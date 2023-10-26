import numpy as np


class State:
    similarity_weight = 0.5

    def __init__(self, obs, action, next_obs, reward):
        self.obs = obs
        self.action = action
        self.next_obs = next_obs
        self.reward = reward

    def compare(self, other):
        dot = np.dot(self.obs, other.obs)
        self_obs_norm = np.linalg.norm(self.obs)
        other_obs_norm = np.linalg.norm(other.obs)
        obs_similarity = dot / (self_obs_norm * other_obs_norm)
        reward_similarity = self.reward - other.reward
        return (
            State.similarity_weight * obs_similarity
            + (1 - State.similarity_weight) * reward_similarity
        )
