import numpy as np
class QL():
    """Q-Learning implementation"""
    def __init__(this, state_dim = 6400, actions = [2, 3]):
        # hyperparameters
        this.name = "DQL"
        this.resume = True
        this.eta = .628
        this.gma = .9
        this.rev_list = []
        this.total_reward = 0
        this.actions = actions
        this.Q = np.zeros([state_dim, len(actions)])
        this.episode_number = 0
        this.s = np.zeros(state_dim, dtype = 'int32')
        this.s1 = np.zeros(state_dim, dtype = 'int32')
        this.a = 0

    def step(this, observation):
        this.s1 = observation.astype('int32')
        this.a = np.argmax(this.Q[this.s,:] + np.random.randn(1, len(this.actions))*(1./(this.episode_number+1)))
        # print(this.a)
        action = this.actions[this.a]

        return action

    def update(this, reward, episode_number):
        this.episode_number = episode_number
        if (reward != 0):
            this.Q[this.s, this.a] = this.Q[this.s,this.a] + this.eta*(reward + this.gma * np.max(this.Q[this.s1,:]) - this.Q[this.s,this.a])
            this.total_reward += reward
        this.s = this.s1
