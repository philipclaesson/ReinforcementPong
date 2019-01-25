## Installation
- run `pip3 install gym`
- run `pip3 install gym[atari]`

## Run
`python3 pong.py`

## Good to know
- round = a game sequence that ends with the ball going past one of the players. One point is awarded to the winner and rewards (+1 and -1) is given.
- episode = 20 rounds. After an episode the scores are reset to 0 - 0.

## About
*Pong.py* is the main file which loads a model of your choice, loads the pong environment. It interacts through any imported model through:
- model.step() to get a sampled action from the policy
- model.update() to return an array of rewards.

Pong.py calls model.update() _after each episode_. In combination with a default batch size of 20, this means that the model is effectively updated every 20*10 = 200 rounds.

The file also contains some bookkeeping, in order to keep track of scores and _after which round out model beats the OpenAI-bot for the first time_.

*Pong-roundwise.py* is an attempt to speed up convergence, we call model.update() for every _round_ instead of every _episode_.

*PolNet* is the first model, a policy network as developed by Karpathy.

