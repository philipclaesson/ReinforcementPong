""" Trains a model to play pong using Q-Learning. Uses OpenAI Gym. """
import numpy as np
import _pickle as pickle
import gym
import time
import QL



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
render = True

# Create a new QL
model = QL.QL()

# Load the episode number if we decide to restart training (because we want to know how many episodes we have trained right.)
if model.resume:
    try:
        episode_number = pickle.load(open('{}_episode_number.p'.format(model.name), 'rb'))
    except:
        episode_number = 0
    print("Episode number initialized as {}".format(episode_number))

our_score = 0
their_score = 0
wins = 0
losses = 0
first_ep_win = -1
win_counter = []

while True:
  if render: env.render()
  if delay: time.sleep(delay)
  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  # send the state to the model to get an action
  action = model.step(cur_x)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  # feed the episode data to the model for policy update
  # if (reward != 0):
  model.update(reward, episode_number)

  if done: # an episode finished (a game of 20 was either lost or won)
    episode_number += 1

    if (episode_number % 100 == 0):
        pickle.dump(episode_number, open('{}_episode_number.p'.format(model.name), 'wb'))

    # was this the first episode win?
    if (wins == 20 and first_ep_win == -1):
        first_ep_win = episode_number
        print("Won the first game after {} episodes. ".format(first_ep_win))
        pickle.dump(first_ep_win, open('{}_first_ep_win.p'.format(model.name), 'wb'))
        pickle.dump(win_counter, open('{}_win_counter.p'.format(model.name), 'wb'))

    # reset
    wins, losses = 0, 0

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    reward_sum = 0
    observation = env.reset() # reset env
