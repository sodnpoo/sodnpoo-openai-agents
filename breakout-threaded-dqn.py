import tensorflow as tf
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense, Dropout, Flatten, Reshape, LSTM, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.callbacks import LearningRateScheduler
from keras.constraints import maxnorm
from gym import wrappers
from statistics import mean

import cv2

from queue import Queue
import threading

EPISODES = 1000000
#MEMORY_FRAMES = 4
MEMORY_FRAMES = 8
MEMORY_MAX = 50000
MEMORY_SAMPLE = 1000
FRAME_W = 80
FRAME_H = 80

learning_rate = 0.0001

# hideous global for the TF reference for multithreading
graph = None

class DQNAgent(threading.Thread):
    def stop(self):
        self.running = False

    # training thread
    def run(self):
        while self.running:
            batch = self.train_queue.get()
            self.train_batch(batch)

    def __init__(self, state_size, action_size):
        threading.Thread.__init__(self)
        self.running = True

        self.render = True

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_decay_start = 20000
        self.epsilon_min = 0.1
        self.batch_size = 32
        self.train_start = 20000
        # create replay memory using deque
        self.memory = []

        # create main model
        self.model = self.build_model()

        # training queue for the training thread
        self.train_queue = Queue(maxsize=32)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(24, (3, 3), activation='relu', input_shape=(MEMORY_FRAMES, 80, 80), data_format='channels_first'))
        model.add(MaxPooling2D((2, 2), data_format='channels_first'))
        model.add(Conv2D(32, (3, 3), activation='relu', data_format='channels_first'))
        model.add(MaxPooling2D((2, 2), data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        adam = Adam(lr=0.0001)
        model.compile(loss='mse', optimizer=adam)
        model.summary()

        # needed to use TF+keras in multiple threads
        model._make_predict_function()
        model._make_train_function()
        self.graph = tf.get_default_graph()

        return model

    def get_action(self, state):
        # sometimes return a random action to help exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # ask the model for the next action
        q_value = self.model.predict(state)
        action = np.argmax(q_value[0])
        return action

    def decay_epsilon(self):
        if len(self.memory) < self.epsilon_decay_start:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # save sample to the replay memory
    def append_sample(self, state, action, reward, next_state, done, dead):
        self.memory.append((state, action, reward, next_state, done, dead))

    # randomly sample the replay memory
    def memory_sample(self, count):
        if len(self.memory) < self.train_start:
            return []

        memory = list(self.memory)
        batch_size = min(count, len(memory))
        mini_batch = random.sample(memory, batch_size)
        return mini_batch

    # enqueue a batch of data to train
    def train_batch_queue(self, batch):
        self.train_queue.put(batch)

    def train_batch(self, batch):
        num = int(len(batch) / 1024) + 1
        #print("num: %d" % num)
        for mini in np.array_split(batch, num):
            self.train_mini_batch(mini)

    def create_dqn_batches(self, batch):
        batch_size = len(batch)

        update_input = np.zeros((batch_size, MEMORY_FRAMES, 80, 80))
        update_target = np.zeros((batch_size, MEMORY_FRAMES, 80, 80))
        action, reward, done, dead = [], [], [], []

        for i in range(batch_size):
            update_input[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            update_target[i] = batch[i][3]
            done.append(batch[i][4])
            dead.append(batch[i][5])

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)

        return update_input, target, target_val, reward, action, dead

    def train_mini_batch(self, batch):

        #global graph
        with self.graph.as_default():

            def file_backed_lrate(epoch):
                lrate = learning_rate
                with open('lrate.txt', 'r') as f:
                    lrate = f.read().rstrip()

                lrate = float(lrate)
                #print("lrate: %.9f" % lrate)
                return lrate
            lrate = LearningRateScheduler(file_backed_lrate)
            callbacks_list = [lrate]

            update_input, target, target_val, reward, action, dead = self.create_dqn_batches(batch)

            for i in range(len(batch)):
                # Q Learning: get maximum Q value at s' from model
                if dead[i]:#done[i]:
                    target[i][action[i]] = reward[i]
                else:
                    target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_val[i]))

            # and train...
            self.model.fit(
                update_input,
                target,
                batch_size=self.batch_size,
                epochs=1,
                verbose=0,
                callbacks=callbacks_list)

# crop and shrink the playarea to minimise the amount of data
def downsample(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    x = 8
    y = 32
    w = image.shape[1] - (x*2)
    h = image.shape[0] - y - 16
    cropped = image[y:y+h, x:x+w]
    ret, cropped = cv2.threshold(cropped, 4, 255, cv2.THRESH_BINARY)
    image = cv2.resize(cropped, (80, 80), )
    return image

# ensure we have MEMORY_FRAMES x states in the expected shape for the model
def stack_state(state, states):
    ds_state = downsample(state)
    states.append(ds_state)
    stacked_states = np.stack(states, axis=0)
    stacked_states = np.reshape(stacked_states, [1, MEMORY_FRAMES, 80, 80])
    return stacked_states

if __name__ == "__main__":
    env = gym.make('Breakout-v4')
    env.frameskip = 4

    state_size = FRAME_H*FRAME_W
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    agent.start()

    scores, episodes = [], []

    logf = open("log.csv", 'w')
    print("episode,step,score,avg_score,epsilon", file=logf)

    acc_score = 0.0
    last_x_scores = deque(maxlen=1000)

    for e in range(EPISODES):
        states = deque(maxlen=MEMORY_FRAMES)
        next_states = deque(maxlen=MEMORY_FRAMES)
        state = env.reset()

        done = False
        score = 0
        steps = 0
        lives = 5
        moves = []

        max_fire = 200
        fire_counter = max_fire

        # make sure the states/next_states buffers are full
        ds_state = downsample(state)
        for i in range(0, MEMORY_FRAMES):
            states.append(ds_state)
            next_states.append(ds_state)

        while not done:
            if agent.render:
                env.render()

            # stack the states
            stacked_states = stack_state(state, states)

            action = agent.get_action(stacked_states)
            agent.decay_epsilon()

            # HACK: we need to press fire to start the ball moving
            # this overrides the action if fire counter is active
            # (the model should learn this)
            if fire_counter is not None:
                fire_counter -= 1
                if fire_counter < 0:
                    action = 1 #fire

            # action is fire, we can disable the fire counter
            if action == 1: # fire
                fire_counter = None

            next_state, reward, done, info = env.step(action)

            dead = info['ale.lives']<lives
            lives = info['ale.lives']
            score += reward

            # scale the reward to be further away from zero
            if reward > 0:
                reward = 100 * reward
            # large negative reward if we died
            if dead:
                reward = -1000

            # stack the next states
            stacked_next_states = stack_state(next_state, next_states)

            moves.append((stacked_states, action, reward, stacked_next_states, done, dead))

            state = next_state

            steps += 1

            # we just died, start the fire counter
            if dead and lives > 0:
                fire_counter = max_fire

        if done:
            # calculate the average score over the last X episodes
            last_x_scores.append(score)
            avg_score = mean(last_x_scores)

            # quick and dirty CSV writing
            print("%d,%d,%d,%.9f,%.9f" % (e, steps, score, avg_score, agent.epsilon), file=logf)
            logf.flush()

            # create the training batch and save this episodes moves to the replay memory
            batch = agent.memory_sample(MEMORY_SAMPLE)
            if len(agent.memory) < MEMORY_MAX or score > avg_score:
                moves = moves[-MEMORY_SAMPLE:] # only use the last 1000 moves, ie. the end of the game
                batch = moves + batch

                # save to the replay memory
                for move in moves:
                    _stacked_states, _action, _reward, _stacked_next_states, _done, _dead = move
                    agent.append_sample(_stacked_states, _action, _reward, _stacked_next_states, _done, _dead)

            # enqueue the batch
            agent.train_batch_queue(batch)

            print("episode:", e, "  score:", score, "  memory length:", len(agent.memory), "  epsilon:", agent.epsilon)

            # limit the size of the replay memory
            agent.memory = agent.memory[-MEMORY_MAX:]

            # save the model every 1000 episodes
            if e % 1000 == 0:
                print("saving model...")
                agent.model.save('./breakout-threaded-dqn.h5')
