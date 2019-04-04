import gym
import numpy as np
from collections import deque
from gym import wrappers
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
import cv2
import random
import time

# imports for multithreading
from threading import Thread
from queue import Queue
import tensorflow as tf

##### breakout DQN
start_timestamp = int(time.time())

EPISODES = 10000            # number of episodes to play
MEMORY_FRAMES = 8           # number of frames to stack
REPLAY_MEMORY = 20000       # amount of moves to remember in the replay memory
                            # 20k will use ~4GB RAM
FORCE_FIRE = True           # force the fire button to be pressed after dying
                            # if the model doesn't
DEBUG = False               # set to True for lots of debug output
MODEL_FILENAME = "dqn.%d.model.h5" % start_timestamp
                            # filename to save the model to
SAVE_EVERY_EPISODE = 100    # save the model every 100 episodes
RENDER = True               # render the Atari window
MULTITHREADING = True       # use a background thread to do the training
                            # this will be faster, but you'll need to double
                            # CTRL+C to quit
TRAINING_QUEUE_LENGTH = 32  # size of the batch fifo buffer
CSV_LOG_FILENAME = "dqn.%d.csv" % start_timestamp
                            # filename to log metrics to

class Agent:
    def __init__(self, env):
        self.actions = env.unwrapped.get_action_meanings()
        self.action_size = env.action_space.n

        self.states = deque(maxlen=MEMORY_FRAMES)

        # number of frame to wait after dying before pressing fire
        self.fire_steps = 150
        self.fire_counter = self.fire_steps

        # initial learning rate
        self.learning_rate = 0.001

        # DQN discount factor
        self.discount_factor = 0.99

        # initial epsilon value
        self.epsilon = 1.0
        # epsilon decay multiplier
        self.epsilon_decay = 0.9999
        # minimum amount of replay memory before we start to decay the epsilon
        self.epsilon_decay_min_memory = REPLAY_MEMORY
        # minimum epsilon value
        self.epsilon_min = 0.1

        # training batch size
        self.batch_size = 32

        # minimum amount of replay memory before we start training
        self.min_training_memory = REPLAY_MEMORY

        # replay memory
        # should be a deque but sampling a deque is slow
        self.memory = []
        # number previous moves to remember
        self.max_memory = REPLAY_MEMORY

        # build model
        self.model = self.build_model()

    # decay the epsilon value until it's at it's minimum
    def decay_epsilon(self):
        if len(self.memory) < self.epsilon_decay_min_memory:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # randomly sample the replay memory
    def memory_sample(self, count):
        memory = list(self.memory)
        batch_size = min(count, len(memory))
        batch = random.sample(memory, batch_size)
        return batch

    # take the tail end of the replay memory
    def tail_memory(self):
        self.memory = self.memory[-self.max_memory:]

    def memory_length(self):
        return len(self.memory)

    # save sample to the replay memory
    def append_sample(self, state, action, reward, next_state, done, dead):
        self.memory.append((state, action, reward, next_state, done, dead))
        self.tail_memory()

    # keras model
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

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        model.summary()

        return model

    # return the next action based on the stack of states
    def action(self, stack):
        # sometimes return a random action to help exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # predict the action probabilities
        probs = self.model.predict(stack)
        # pick the most likely
        action = np.argmax(probs[0])

        # force fire if we died and we're sitting there
        if FORCE_FIRE:
            action = self.force_fire(action)

        return action

    # wind down the counter, and return fire action when it expires
    def force_fire(self, action):
        if self.fire_counter is not None:
            self.fire_counter -= 1
            if self.fire_counter < 0:
                action = self.actions.index('FIRE')

        # fire pressed, disable the counter
        if action == self.actions.index('FIRE'):
            self.fire_counter = None

        return action

    # downsample (crop, black and white) the state
    def downsample(self, image):
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
    def stack_state(self, state, states):
        ds_state = self.downsample(state)

        #make sure we have MEMORY_FRAMES number of states
        while len(states) < MEMORY_FRAMES-1:
            states.append(ds_state)

        states.append(ds_state)
        stacked_states = np.stack(states, axis=0)
        stacked_states = np.reshape(stacked_states, [1, MEMORY_FRAMES, 80, 80])
        return stacked_states

    # start the fire counter running
    def died(self):
        self.fire_counter = self.fire_steps

    def scale_reward(self, reward, dead):
        # scale the reward to be further away from zero
        if reward > 0:
            reward = 100 * reward
        # large negative reward if we died
        if dead:
            reward = -1000
        return reward

    # predict the training targets for DQN
    def predict_targets(self, update_input, update_target):
        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)
        return target, target_val

    # split the training data up into arrays for training
    def create_dqn_batches(self, batch):
        batch_size = len(batch)

        update_input = np.zeros((batch_size, MEMORY_FRAMES, 80, 80))
        update_target = np.zeros((batch_size, MEMORY_FRAMES, 80, 80))
        action, reward, done, dead = [], [], [], []

        for i in range(batch_size):
            update_input[i] = batch[i][0]
            update_target[i] = batch[i][3]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            done.append(batch[i][4])
            dead.append(batch[i][5])

        return update_input, update_target, reward, action, dead

    def train(self, batch):
        # DQN magic: this essentially allows rewards to flow backwards
        # from the point they were awarded to the moves before them
        update_input, update_target, reward, action, dead = self.create_dqn_batches(batch)

        # we predict what the output will be for this state and the next state
        target, target_val = self.predict_targets(update_input, update_target)

        # we train the model to use the reward value for the 'best' action
        # if we didn't die, then we also add the next state result multiplied
        # by the discount factor - this allows the reward to flow back
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
            callbacks=[])

class ThreadedAgent(Thread, Agent):
    # extends the Agent to run the training in a background thread
    def __init__(self, env):
        Thread.__init__(self)
        Agent.__init__(self, env)

        self.train_queue = Queue(maxsize=TRAINING_QUEUE_LENGTH)

        self.thread_model()

    # set up the keras model to work with threads
    def thread_model(self):
        # these are normally run lazily, this create problems with threads
        self.model._make_predict_function()
        self.model._make_train_function()
        self.graph = tf.get_default_graph()

    def stop(self):
        print("stopping training thread.")
        self.running = False

    # training thread
    def run(self):
        print("starting training thread...")
        self.running = True
        while self.running:
            batch = self.train_queue.get()
            Agent.train(self, batch)

    def train(self, batch):
        self.train_queue.put(batch)

if __name__ == "__main__":
    env = gym.make('Breakout-v4')
    env.frameskip = 4

    agent = None
    if MULTITHREADING:
        agent = ThreadedAgent(env)
        agent.start()
    else:
        agent = Agent(env)

    if CSV_LOG_FILENAME is not None:
        # open csv log and write header
        print("logging to '%s' ..." % CSV_LOG_FILENAME)
        logf = open(CSV_LOG_FILENAME, 'w')
        print("episode,steps,score,memory,epsilon", file=logf)

    for e in range(EPISODES):
        done = False
        score = 0
        steps = 0
        lives = 5

        # record this episode's moves
        moves = []

        # states and next states buffers
        states = deque(maxlen=MEMORY_FRAMES)
        next_states = deque(maxlen=MEMORY_FRAMES)

        state = env.reset()
        while not done:
            if RENDER:
                env.render()

            # stack state 0..X
            stack = agent.stack_state(state, states)

            action = agent.action(stack)
            agent.decay_epsilon()

            next_state, reward, done, info = env.step(action)

            # stack state 1..X+1
            next_stack = agent.stack_state(next_state, next_states)

            score += reward
            dead = info['ale.lives']<lives
            lives = info['ale.lives']

            scaled_reward = agent.scale_reward(reward, dead)

            # save the moves
            move = stack, action, scaled_reward, next_stack, done, dead
            moves.append(move)

            if dead:
                agent.died()

            if DEBUG:
                print("episode: %03d\tstep: %03d\tscore: %03d\treward: %d\tlives: %d\tdead: %s\tdone: %s" % (e, steps, score, reward, lives, str(dead)[0], str(done)[0]))

            state = next_state
            steps += 1

        # do a training phase here.

        # sample as many moves from the replay memory as we've just done
        sample = agent.memory_sample(len(moves))
        # add the current moves to the head of the replay memory
        for move in moves:
            state, action, scaled_reward, next_state, done, dead = move
            agent.append_sample(state, action, scaled_reward, next_state, done, dead)

        # train the model on both the replay samples and moves
        training_batch = sample + moves
        agent.train(training_batch)

        print("episode: %05d\tsteps: %04d\tscore: %03d\tmemory: %06d\tepsilon: %.2f" % (e, steps, score, agent.memory_length(), agent.epsilon))

        if CSV_LOG_FILENAME is not None:
            print("%d,%d,%d,%d,%.2f" % (e, steps, score, agent.memory_length(), agent.epsilon), file=logf)
            logf.flush()

        # save the model every 1000 episodes
        if e % SAVE_EVERY_EPISODE == 0:
            print("saving model (%s)..." % MODEL_FILENAME)
            agent.model.save(MODEL_FILENAME)

    if CSV_LOG_FILENAME is not None:
        logf.close()

    if MULTITHREADING:
        agent.stop()

    env.close()
