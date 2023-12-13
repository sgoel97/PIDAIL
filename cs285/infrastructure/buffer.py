import numpy as np
import random


class ReplayBuffer:
    def __init__(self, capacity=10000, device="cpu"):
        self.capacity = capacity
        self.curr_size = 0
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None
        self.device = device

    def __len__(self):
        return self.curr_size

    def insert(self, obs, action, reward, next_obs, done):
        if type(reward) == float or type(reward) == int:
            reward = np.array(reward)
        if type(done) == bool:
            done = np.array(done)
        if type(action) == int:
            action = np.array(action, dtype=np.int64)

        if self.observations is None:
            self.observations = np.empty((self.capacity, *obs.shape), dtype=obs.dtype)
            self.actions = np.empty((self.capacity, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.capacity, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.empty(
                (self.capacity, *next_obs.shape), dtype=next_obs.dtype
            )
            self.dones = np.empty((self.capacity, *done.shape), dtype=done.dtype)

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
        rand_indices = np.random.randint(0, self.curr_size, size=amount) % self.capacity
        return {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "dones": self.dones[rand_indices],
        }


class PrioritizedReplayBuffer:  # FIXME: could make it extend from ReplayBuffer somehow?
    e = 0.0  # revisit
    a = 0.6
    beta = 0.0
    beta_inc = 0.001

    def __init__(self, capacity = 1000000):
        self.sum_tree = SumTree(capacity)
        self.capacity = capacity
    
    def get_priority(self, error):
        return (error + self.e) ** self.a
    
    def insert(self, obs, action, reward, next_obs, done, demo_info, error = None):
        if error is None:
            priority = self.sum_tree.tree[0]
            if priority == 0:
                priority = 0.1
            else:
                priority = self.sum_tree.get(priority * 0.9)[1]
        else:
            priority = self.get_priority(error)
        self.sum_tree.insert(priority, obs, action, reward, next_obs, done, demo_info)
    
    def sample(self, amount):
        sample = {"observations": [], "actions": [], "rewards": [], "next_observations": [], "dones": [], "demo_infos": []}
        indices = []
        segment = self.sum_tree.total() / amount
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_inc])

        for i in range(amount):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.sum_tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            for key in data:
                sample[key].append(data[key])
        
        probs = np.array(priorities) / self.sum_tree.total()
        weights = np.power(self.sum_tree.curr_size * probs, -self.beta)
        weights /= weights.max()
        return sample, indices, weights
    
    def update(self, idx, error):
        priority = self.get_priority(error)
        self.sum_tree.update(idx, priority)


class SumTree:
    write = 0

    def __init__(self, capacity = 1000000):
        self.capacity = capacity
        self.curr_size = 0
        self.start = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.observations = [None for _ in range(capacity)]
        self.actions = [None for _ in range(capacity)]
        self.rewards = [None for _ in range(capacity)]
        self.next_observations = [None for _ in range(capacity)]
        self.dones = [None for _ in range(capacity)]
        self.demo_infos = [None for _ in range(capacity)]
    
    def propagate(self, idx, delta):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self.propagate(parent, delta)
    
    def retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self.retrieve(left, s)
        return self.retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]
    
    def insert(self, priority, obs, action, reward, next_obs, done, demo_info):
        idx = self.write + self.capacity - 1
        self.observations[self.write] = obs
        self.actions[self.write] = action
        self.rewards[self.write] = reward
        self.next_observations[self.write] = next_obs
        self.dones[self.write] = done
        self.demo_infos[self.write] = demo_info
        
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = self.start
        if self.curr_size < self.capacity:
            self.curr_size += 1
    
    def update(self, idx, priority):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self.propagate(idx, delta)
    
    def get(self, s):
        tree_idx = self.retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        data = {
            "observations": self.observations[data_idx],
            "actions": self.actions[data_idx],
            "rewards": self.rewards[data_idx],
            "next_observations": self.next_observations[data_idx],
            "dones": self.dones[data_idx],
            "demo_infos": self.demo_infos[data_idx]
        }
        return tree_idx, self.tree[tree_idx], data


if __name__ == "__main__":
    # Testing PRB
    prb = PrioritizedReplayBuffer(10)
    prb.insert([21], 0, 10, [42], True, None, 1)
    prb.insert([42], 1, 20, [69], True, None, 10)
    prb.insert([69], 2, 30, [420], True, None, 100)
    data, idx, _ = prb.sample(2)
    print(prb.sample(5)[2])
    prb.update(11, 0)
    print(prb.sample(5)[2])