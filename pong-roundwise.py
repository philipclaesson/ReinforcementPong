""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
    difference from the original Pong.py is that we update for every round instead of every episode
    (an episode consists of 20 games.)"""
import numpy as np
import _pickle as pickle
import gym
import time
import PolNet



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
round_number = 0
episode_number = 0
rewards = []
render = False

# Create a new polnet
model = PolNet.PolNet()
model.name = "Karp_PolNet_rw"

our_score = 0
their_score = 0
wins = 0
losses = 0
first_ep_win = -1
win_counter = []

# Load the episode number if we decide to restart training (because we want to know how many episodes we have trained right.)
if model.resume:
    try:
        episode_number = pickle.load(open('{}_episode_number_rw '.format(model.name), 'rb'))
    except:
        episode_number = 0
    print("Episode number initialized as {}".format(episode_number))


while True:
  if render: env.render()
  if delay: time.sleep(delay)
  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(model.D)
  prev_x = cur_x

  # send the state to the model to get an action
  action = model.step(x)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  # If someone won the round, keep track of points.
  if (reward):
      our_score += 1
      wins += 1
  elif (reward == -1):
      their_score += 1
      losses +=1

# record reward (has to be done after we call step() to get reward for previous action)
  rewards.append(reward)


  # if the round is over (someone scored, so reward is -1 or 1)
  if (reward != 0):
    round_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    round_rewards = np.vstack(rewards)
    rewards = [] # reset array memory

    # feed the episode data to the model for policy update
    model.update(round_rewards, round_number)


  if done: # an episode finished (a game of 20 was either lost or won)
    episode_number += 1

    # keep track of how many rounds we won this episode
    win_counter.append(wins)

    # was this the first episode win?
    if (wins == 20 and first_ep_win == -1):
        first_ep_win = episode_number
        print("Won the first game after {} episodes. ".format(first_ep_win))
        pickle.dump(first_ep_win, open('{}_first_ep_win_rw'.format(model.name), 'wb'))
        pickle.dump(win_counter, open('{}_win_counter_rw'.format(model.name), 'wb'))

    # reset
    wins, losses = 0, 0

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  #if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    #print (("ep {}: game finished, reward: {}".format(episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
