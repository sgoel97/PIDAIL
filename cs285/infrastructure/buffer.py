class ReplayBuffer:
    def __init__(self, capacity = 10000):
        self.capacity = capacity
        self.curr_size = 0
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None
    
    
    def __len__(self):
        return self.curr_size
    
    
    def insert(self, obs, action, reward, next_obs, done):
        if type(reward) == float or type(reward) == int:
            reward = np.array(reward)
        if type(done) == bool:
            done = np.array(done)
        if type(action) == int:
            action = np.array(action, dtype = np.int64)
        
        if self.observations is None:
            self.observations = np.empty((self.capacity, *obs.shape), dtype = obs.dtype)
            self.actions = np.empty((self.capacity, *action.shape), dtype = action.dtype)
            self.rewards = np.empty((self.capacity, *reward.shape), dtype = reward.type)
            self.next_observations = np.empty((self.capacity, *next_obs.shape), dtype = next_obs.dtype)
            self.dones = np.empty((self.capacity, *done.shape), dtype = done.dtype)
        
        assert obs.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        assert reward.shape == ()
        assert next_obs.shape == self.next_observations.shape[1:]
        assert done.shape == ()

        self.observations[self.curr_size % self.capacity] = obs
        self.actions[self.curr_size % self.capacity] = action
        self.rewards[self.curr_size % self.capacity] = reward
        self.next_observations[self.curr_size % self.capacity] = next_obs
        self.dones[self.curr_size % self.capacity] = done
        self.curr_size += 1
    
    
    def sample(self, amount):
        rand_indices = np.random.randint(0, self.curr_size, size = amount) % self.capacity
        return {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "dones": self.dones[rand_indices],
        }