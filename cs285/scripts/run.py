import gymnasium as gym
import torch.nn as nn
from cs285.networks.mlp import MLP

env = gym.make("CartPole-v1")
observation, info = env.reset()

action_space = env.action_space.n
obs_dim = env.observation_space.shape[0]

loss_fn = nn.MSELoss()
# agent = MLP(obs_dim=obs_dim, hidden_dims=[512, 216, 64], action_dim=action_space, loss_fn=loss_fn)

for _ in range(1000):
    # action = agent(observation)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # agent.update(observation, action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
