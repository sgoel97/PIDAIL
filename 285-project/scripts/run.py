import sys
sys.path.append("/Users/tutrinh/Academic/CS285/285-project/285-project/")

import gymnasium as gym
from infrastructure.buffer import ReplayBuffer
from infrastructure.utils import *
from agents.agent import Agent
import matplotlib.pyplot as plt
from tqdm import tqdm


env = gym.make("CartPole-v1")
agent = Agent(env.observation_space.shape, env.action_space.n)
replay_buffer = ReplayBuffer()
total_steps = 1000
non_learning_steps = 50
batch_size = 32

losses = []
q_values = []
target_values = []

observation, _ = env.reset()
for i in tqdm(range(total_steps)):
    action = agent.get_action(observation)
    next_observation, reward, terminated, truncated, _ = env.step(action)
    replay_buffer.insert(observation, action, reward, next_observation, terminated and not truncated)

    if terminated or truncated:
        observation, _ = env.reset()
    else:
        observation = next_observation
    
    if i > non_learning_steps:
        experience = from_numpy(replay_buffer.sample(batch_size))
        update_info = agent.update(batch["observations"], batch["actions"], batch["rewards"], batch["next_observations"], batch["dones"])
        losses.append(update_info["q_net_loss"])
        q_values.append(update_info["q_value"])
        target_values.append(update_info["target_value"])
    else:
        losses.append(0)
        q_values.append(0)
        target_values.append(0)

agent.q_net.save("../networks/q_net.pt")
agent.target_net.save("../networks/target_net.pt")

plt.plot(range(total_steps), losses, label = "Loss")
plt.plot(range(total_steps), q_values, label = "Q-value")
plt.plot(range(total_steps), target_values, label = "Target value")
plt.legend()
plt.show()

env.close()
