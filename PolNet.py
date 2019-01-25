import numpy as np
class PolNet():
    "Karpathys implementation of a policy network"
    def __init__(this):
        # hyperparameters
        this.name = "Karp_PolNet"
        this.H = 200 # number of hidden layer neurons
        this.batch_size = 10 # every how many episodes to do a param update?
        this.learning_rate = 1e-4
        this.gamma = 0.99 # discount factor for reward
        this.decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
        this.resume = False

        # model initialization
        this.D = 80 * 80 # input dimensionality: 80x80 grid

        # this.xs, this.hs, this.dlogps, this.drs = [],[],[],[]
        this.observations, this.hidden_states, this.dlogps = [],[],[]

        if this.resume:
          this.model = pickle.load(open('save.p', 'rb'))

        else:
          this.model = {}
          this.model['W1'] = np.random.randn(this.H,this.D) / np.sqrt(this.D) # "Xavier" initialization
          this.model['W2'] = np.random.randn(this.H) / np.sqrt(this.H)


        this.grad_buffer = { k : np.zeros_like(v) for k,v in this.model.items() } # update buffers that add up gradients over a batch
        this.rmsprop_cache = { k : np.zeros_like(v) for k,v in this.model.items() } # rmsprop memory

    def step(this, observation):
        """ takes a step, returns an action """
        # forward the policy network and sample an action from the returned probability
        aprob, hidden = this.policy_forward(observation)
        action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

        # record various intermediates (needed later for backprop)
        this.observations.append(observation) # observation
        this.hidden_states.append(hidden) # hidden state
        y = 1 if action == 2 else 0 # a "fake label"
        this.dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        return action


    def update(this, rewards, episode_number):
        """updates the policy based on the rewards of the previous episode"""
        epx = np.vstack(this.observations)
        eph = np.vstack(this.hidden_states)
        epdlogp = np.vstack(this.dlogps)

        # reset arrays
        this.observations, this.hidden_states, this.dlogps = [],[],[]

        # compute the discounted reward backwards through time
        discounted_epr = this.discount_rewards(rewards)

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = this.policy_backward(epx, eph, epdlogp)
        for k in this.model: this.grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % this.batch_size == 0:
          for k,v in model.items():
            g = grad_buffer[k] # gradient
            this.rmsprop_cache[k] = decay_rate * this.rmsprop_cache[k] + (1 - decay_rate) * g**2
            this.model[k] += learning_rate * g / (np.sqrt(this.rmsprop_cache[k]) + 1e-5)
            this.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # save model to file
        if episode_number % 100 == 0: pickle.dump(this.model, open('{}_model.p'.format(this.name), 'wb'))


    def discount_rewards(this, r):
      """ take 1D float array of rewards and compute discounted reward """
      discounted_r = np.zeros_like(r)
      running_add = 0
      for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * this.gamma + r[t]
        discounted_r[t] = running_add
      return discounted_r

    def policy_forward(this, x):
      h = np.dot(this.model['W1'], x)
      h[h<0] = 0 # ReLU nonlinearity
      logp = np.dot(this.model['W2'], h)
      p = this.sigmoid(logp)
      return p, h # return probability of taking action 2, and hidden state

    def policy_backward(this, epx, eph, epdlogp):
      """ backward pass. (eph is array of intermediate hidden states) """
      dW2 = np.dot(eph.T, epdlogp).ravel()
      dh = np.outer(epdlogp, this.model['W2'])
      dh[eph <= 0] = 0 # backpro prelu
      dW1 = np.dot(dh.T, epx)
      return {'W1':dW1, 'W2':dW2}

    def sigmoid(this, x):
        return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]




