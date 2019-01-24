class PolNet():
    "Karpathys implementation of a policy network"
    def __init__():
        # hyperparameters
        this.H = 200 # number of hidden layer neurons
        this.batch_size = 10 # every how many episodes to do a param update?
        this.learning_rate = 1e-4
        this.gamma = 0.99 # discount factor for reward
        this.decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
        this.render = True
        this.resume = False

        this.grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
        this.rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

    if resume:
      model = pickle.load(open('save.p', 'rb'))
    else:
      model = {}
      model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
      model['W2'] = np.random.randn(H) / np.sqrt(H)

      
    def discount_rewards(r):
      """ take 1D float array of rewards and compute discounted reward """
      discounted_r = np.zeros_like(r)
      running_add = 0
      for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
      return discounted_r

    def policy_forward(x):
      h = np.dot(model['W1'], x)
      h[h<0] = 0 # ReLU nonlinearity
      logp = np.dot(model['W2'], h)
      p = sigmoid(logp)
      return p, h # return probability of taking action 2, and hidden state

    def policy_backward(eph, epdlogp):
      """ backward pass. (eph is array of intermediate hidden states) """
      dW2 = np.dot(eph.T, epdlogp).ravel()
      dh = np.outer(epdlogp, model['W2'])
      dh[eph <= 0] = 0 # backpro prelu
      dW1 = np.dot(dh.T, epx)
      return {'W1':dW1, 'W2':dW2}

    def discount_rewards(r):
      """ take 1D float array of rewards and compute discounted reward """
      discounted_r = np.zeros_like(r)
      running_add = 0
      for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
      return discounted_r

    def policy_forward(x):
      h = np.dot(model['W1'], x)
      h[h<0] = 0 # ReLU nonlinearity
      logp = np.dot(model['W2'], h)
      p = sigmoid(logp)
      return p, h # return probability of taking action 2, and hidden state

    def policy_backward(eph, epdlogp):
      """ backward pass. (eph is array of intermediate hidden states) """
      dW2 = np.dot(eph.T, epdlogp).ravel()
      dh = np.outer(epdlogp, model['W2'])
      dh[eph <= 0] = 0 # backpro prelu
      dW1 = np.dot(dh.T, epx)
      return {'W1':dW1, 'W2':dW2}



