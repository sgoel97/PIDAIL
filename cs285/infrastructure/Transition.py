import numpy as np


class Transition:
    similarity_weight = 0.5

    def __init__(self, obs, action, next_obs, reward, done):
        self.obs = obs
        self.action = action
        self.next_obs = next_obs
        self.reward = reward
        self.done = done

    def compare(self, other):
        dot = np.dot(self.obs, other.obs)
        self_obs_norm = np.linalg.norm(self.obs)
        other_obs_norm = np.linalg.norm(other.obs)
        obs_similarity = dot / (self_obs_norm * other_obs_norm)
        reward_similarity = abs(self.reward - other.reward)
        return (
            Transition.similarity_weight * obs_similarity
            + (1 - Transition.similarity_weight) * reward_similarity
        )
