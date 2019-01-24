""" Uses OpenAI Gym's implementation of Pong. """
import numpy as np
import gym


## Settings ##
render = True

env = gym.make("Pong-v0")
while True:
    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    #cur_x = prepro(observation)
    #x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    #prev_x = cur_x
