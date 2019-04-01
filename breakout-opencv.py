import sys
import gym
import random
import numpy as np
from gym import wrappers
import cv2

EPISODES = 10       # number of episodes(games) to play
DEBUG = False       # set to True for lots of debug output
DEBUG_SCALE = 3     # how much to scale the debug image video by
OUTPUT_DIR = None   # disables writing video/meta
#OUTPUT_DIR = './'  # write sample videos+meta to this location

# scaling wrapper for imshow()
def imshow(label, image, scale=DEBUG_SCALE):
    cv2.imshow(label, cv2.resize(image, (image.shape[1]*scale, image.shape[0]*scale)))

class OpenCvAgent:
    def __init__(self, env):
        self.env = env
        self.actions = env.unwrapped.get_action_meanings()

    def _crop(self, state):
        # crop out the playarea
        x = 8
        y = 32
        w = state.shape[1] - (x*2)
        h = state.shape[0] - y
        return state[y:y+h, x:x+w]

    def _filter(self, state, debug=False):
        # hacked together filter steps to give something findContours() can work with

        steps = []
        state = cv2.Laplacian(state, cv2.CV_8UC1)
        if debug:
            steps.append(state)

        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY);

        state = cv2.blur(state, (6, 6))
        if debug:
            steps.append(cv2.cvtColor(state, cv2.COLOR_GRAY2RGB))

        state = cv2.erode(state, np.ones((3, 3), np.uint8), iterations=3)
        if debug:
            steps.append(cv2.cvtColor(state, cv2.COLOR_GRAY2RGB))

        ret, state = cv2.threshold(state, 4, 255, cv2.THRESH_BINARY)

        if debug:
            steps.append(cv2.cvtColor(state, cv2.COLOR_GRAY2RGB))
            imshow('filtered', np.concatenate(steps, axis=1))

        return state

    def state2coords(self, state, debug=False):
        ball = None, None
        paddle = None

        #opencv uses BGR, convert the state from RGB
        state = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)

        cropped = self._crop(state)
        filtered = self._filter(cropped, debug)

        # find contours in the filtered image
        new_binframe, contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)

            # the ball is 3x3
            if (w, h) == (3, 3):
                ball = x, y
                if debug:
                    cv2.rectangle(cropped, (x,y), (x+w,y+h), (0, 255, 0), 1)

            # the paddle is 157x5
            if (h, y) == (5, 157):
                paddle = x
                if debug:
                    cv2.rectangle(cropped, (x,y), (x+w,y+h), (0, 255, 0), 1)

        if debug:
            imshow('contours', cropped)

        if debug:
            cv2.waitKey(1)

        return ball, paddle

    def action(self, state, debug=False):
        ball, paddle = self.state2coords(state, debug)
        ballx, bally = ball

        if debug:
            print("paddle: %s\tballX: %s\tballY: %s" % (paddle, ballx, bally))

        if paddle == None or ballx == None or bally == None:
            # HACK: we need to press fire to start the ball moving
            # if we can't find the paddle or the ball we return FIRE
            return self.actions.index('FIRE')

        # Try and keep the paddle under the ball
        if paddle > ballx:
            return self.actions.index('LEFT')
        if paddle < ballx:
            return self.actions.index('RIGHT')

        return self.actions.index('NOOP')

if __name__ == "__main__":
    env = gym.make('Breakout-v4')
    env.frameskip = 4

    if OUTPUT_DIR is not None:
        env = wrappers.Monitor(env, directory=OUTPUT_DIR, force=True)

    agent = OpenCvAgent(env)

    for e in range(EPISODES):
        # get the initial game state
        state = env.reset()

        done = False
        score = 0
        step = 0
        lives = 5

        while not done:
            # render the frame
            env.render()

            # ask the agent for the next action based on the current state
            action = agent.action(state, debug=DEBUG)

            # tell the environment to use the action and return the next state,
            # reward, if we're done, and any additional information
            next_state, reward, done, info = env.step(action)

            score += reward # sum the rewards to get the score
            next_lives = info['ale.lives'] # get the number of lives from the info dict
            dead = next_lives<lives # if the lives have gone down, then we lost a life
            lives = next_lives

            if DEBUG:
                print("episode: %03d\tstep: %03d\tscore: %03d\treward: %d\tlives: %d\tdead: %s\tdone: %s" % (e, step, score, reward, lives, str(dead)[0], str(done)[0]))

            state = next_state
            step += 1

        print("episode: %03d\tsteps: %03d\tscore: %03d" % (e, step, score))

    env.close()
