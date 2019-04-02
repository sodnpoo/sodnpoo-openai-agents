import sys
import gym
import numpy as np
from collections import deque
from gym import wrappers
from keras.models import load_model
import cv2

MODEL_FILENAME = 'breakout-pretrained-convnet.h5'
EPISODES = 10
MEMORY_FRAMES = 8
FORCE_FIRE = True
OUTPUT_DIR = None   # disables writing video/meta
#OUTPUT_DIR = './'  # write sample videos+meta to this location


class Agent:
    def __init__(self, env):
        self.actions = env.unwrapped.get_action_meanings()
        self.model = load_model(MODEL_FILENAME)
        self.model.summary()
        self.states = deque(maxlen=MEMORY_FRAMES)
        # number of frame to wait after dying before pressing fire
        self.fire_steps = 150
        self.fire_counter = None

    def action(self, state):
        # stack the frames
        stacked_states = self.stack_state(state, self.states)
        # predict the action probabilities
        probs = self.model.predict(stacked_states)
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

    def died(self):
        self.fire_counter = self.fire_steps

if __name__ == "__main__":
    env = gym.make('Breakout-v4')
    env.frameskip = 4

    if OUTPUT_DIR is not None:
        env = wrappers.Monitor(env, directory=OUTPUT_DIR, force=True)

    agent = Agent(env)

    for e in range(EPISODES):
        done = False
        score = 0
        steps = 0
        lives = 5

        state = env.reset()
        while not done:
            env.render()

            action = agent.action(state)

            next_state, reward, done, info = env.step(action)
            score += reward
            dead = info['ale.lives']<lives
            lives = info['ale.lives']

            if dead:
                agent.died()

            state = next_state
            steps += 1
        print(score)
    env.close()
