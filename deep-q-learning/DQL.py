#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import sys
import random
import numpy as np
from collections import deque
import gym

class DQL:

    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.GAME = "pong"
        self.ACTIONS = 2 # number of valid actions
        self.GAMMA = 0.99 # decay rate of past observations
        self.OBSERVE = 10000. # timesteps to observe before training
        self.EXPLORE = 20000. # frames over which to anneal epsilon
        self.FINAL_EPSILON = 0.0001 # final value of epsilon
        self.INITIAL_EPSILON = 0.1 # starting value of epsilon
        self.REPLAY_MEMORY = 50000 # number of previous transitions to remember
        self.BATCH = 32 # size of minibatch
        self.FRAME_PER_ACTION = 1
        self.D = 80*80
        self.render = True

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    def createNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, self.ACTIONS])
        b_fc2 = self.bias_variable([self.ACTIONS])

        # input layer
        s = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        #h_pool2 = self.max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        #h_pool3 = self.max_pool_2x2(h_conv3)

        #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2

        return s, readout, h_fc1

    def trainNetwork(self, s, readout, h_fc1, sess):
        # define the cost function
        a = tf.placeholder("float", [None, self.ACTIONS])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        # open up a game state to communicate with emulator

        # store the previous observations in replay memory
        D = deque()

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(self.ACTIONS)
        do_nothing[0] = 1
        x_t = self.env.reset()
        r_0 = 0
        terminal = False
        x_t = self.prepro(x_t)
        # x_t = cvtColor(resize(x_t, (80, 80)), COLOR_BGR2GRAY)
        # ret, x_t = threshold(x_t,1,255,THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        # saving and loading networks
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        # start training
        epsilon = self.INITIAL_EPSILON
        t = 0
        while True:
            if self.render:
                self.env.render()
            # choose an action epsilon greedily
            readout_t = readout.eval(feed_dict={s : [s_t]})[0]
            a_t = np.zeros([self.ACTIONS])
            action_index = 0
            if t % self.FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    # print("----------Random Action----------")
                    action_index = random.randrange(self.ACTIONS)
                    a_t[random.randrange(self.ACTIONS)] = 1
                else:
                    action_index = np.argmax(readout_t)
                    a_t[action_index] = 1
            else:
                a_t[0] = 1 # do nothing

            # scale down epsilon
            if epsilon > self.FINAL_EPSILON and t > self.OBSERVE:
                epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE



            # run the selected action and observe next state and reward
            action = np.argmax(a_t) + 2
            # print(action)
            x_t1_colored, r_t, terminal, info = self.env.step(action)
            if terminal:
                self.env.reset()
            # x_t1 = cvtColor(resize(x_t1_colored, (80, 80)), COLOR_BGR2GRAY)
            x_t1 = self.prepro(x_t1_colored)
            # ret, x_t1 = threshold(x_t1, 1, 255, THRESH_BINARY)
            # x_t1 = np.reshape(x_t1, (80, 80, 1))
            #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
            s_t1 = np.append(np.expand_dims(x_t1, axis=2), s_t[:, :, :3], axis=2)

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > self.REPLAY_MEMORY:
                D.popleft()

            # only train if done observing
            if t > self.OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, self.BATCH)
                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + self.GAMMA * np.max(readout_j1_batch[i]))

                # perform gradient step
                train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch}
                )

            # update the old values
            s_t = s_t1
            t += 1

            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(sess, 'saved_networks/' + self.GAME + '-dqn', global_step = t)

            # print info
            state = ""
            if t <= self.OBSERVE:
                state = "observe"
            elif t > self.OBSERVE and t <= self.OBSERVE + self.EXPLORE:
                state = "self.EXPLORE"
            else:
                state = "train"

            if (t % 1000 == 0):
                print("TIMESTEP", t, "/ STATE", state, \
                    "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                    "/ Q_MAX %e" % np.max(readout_t))

    def prepro(self, I):
      """ prepro 210x160x3 uint8 frame into 80x80x1 matrix"""
      I = I[35:195] # crop
      I = I[::2,::2, 0] # downsample by factor of 2
      I[I == 144] = 0 # erase background (background type 1)
      I[I == 109] = 0 # erase background (background type 2)
      I[I != 0] = 1 # everything else (paddles, ball) just set to 1
      return I.astype(np.float)

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)