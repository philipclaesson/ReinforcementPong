## Installation
- run `pip3 install gym`
- run `pip3 install gym[atari]`
You also need tensorflow, numpy etc.. installed. 

## How to run
Run the code by:
- Clone the repo from https://github.com/philipclaesson/ReinforcementPong
- cd into one of the subfolders, i.e. deep-q-learning
- Run python3 pong.py
- By default, the models load the trained models and render the environment graphically on the screen of the user. However, these settings can be changed through modifying the render and resume boolean variables.  

## Background
Reinforcement Learning is one of the hot topics of computer science. It can be considered a field of Machine Learning, where an agent learns to behave in an environment by performing some actions and observing the rewards. There are different kind of problems within RL, all of them with their own variation. For example, in a RL problem the model can be known or unknown, the states may depend on time or not, the states could be partially or fully observable, etc.  In our project, we are tackling one of the basics, yet non-trivial case, of RL, where we are training an agent to play the pong game. We developed different models, with different algorithms, and it is possible to find the code in the Github repository provided.

## Policy Network
We use a policy network, consisting of two fully connected layers. Given an input (the state of the game), the network outputs a stochastic policy in the form of an action (up or down) and a probability of performing the action. We sample an action from this probability and perform the action.

A game is ended when the ball has hit the wall behind one of the players. If our agent wins, we are given +1 point as reward, if our agent loses we are given -1. The network is trained batch-wise through backpropagation.

We tested two different ways of batch wise packprop. In the first, we update our model in batches of 20 games, in the other we update our model after each game. The idea was that the second version would converge faster, however both the models behaved more or less similar after 48h of training.

## Deep Q-learning
Q-learning is one of the most well-known RL algorithms. The name Q-learning comes from the fact that, instead of iterating over the value function V (which depends only from the state s), we are learning directly the quality function Q (which depends also from the action a).
However, in the case of deep q-learning, a neural network is used to learn the matrix Q, thus guaranteeing better results.
In our project, we implemented a CNN that takes as input the image (that is re-scaled and converted to a gray scale) and is able to learn the Q matrix thanks to an epsilon-greedy policy. The idea behind such policy is due to the exploitation-exploration dilemma, where the agent has to choose if it’s more convenient to perform the best known action or to explore new paths. With an epsilon-greedy policy, the model will perform the best action with probability epsilon, while performing the best action with probability one minus epsilon. In our project, we initialized epsilon to a relatively big value at the beginning (to explore new actions) and then decrease it over time.

## Results
After 48h of training, the two policy networks are working well, playing level with the preprogrammed bot. They both receive an average reward of -0.4, and improving.

We have not had the time to train the deep q learning more than a few hours at this point, so we can not really see the full results of it yet. The shallow q-learning algorithm has not been able to learn anything.


![Game Point](https://media.giphy.com/media/cYyf44QXHjpZZ7Kwoz/giphy.gif)
Game point. The green player is controlled by the policy iteration network, which has been trained for some 48h.