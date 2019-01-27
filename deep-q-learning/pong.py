""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import _pickle as pickle
import gym
import time
import DQL
import tensorflow as tf


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

delay = 0 # lil delay. 0.02 makes the game watchable for humans.
env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
running_reward = None
reward_sum = 0
episode_number = 0
rewards = []
render = False

# Create a new polnet
model = DQL.DQL()

our_score = 0
their_score = 0
wins = 0
losses = 0
first_ep_win = -1
win_counter = []

model = DQL.DQL()
sess = tf.InteractiveSession()

s, readout, h_fc1 = model.createNetwork()
model.trainNetwork(s, readout, h_fc1, sess)
