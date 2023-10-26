import sys
# TODO: hacky asf but just append your path to cs285/ here and imports should work
sys.path.append("/Users/tutrinh/Academic/CS285/285-project/cs285/")
import os
pwd = os.getcwd()
if pwd.endswith("285-project"):
    model_path = pwd + "/cs285/networks/"
elif pwd.endswith("cs285"):
    model_path = pwd + "/networks/"
elif pwd.endswith("scripts"):
    model_path = "../networks/"
else:
    raise Exception("Execute run.py from inside either 285-project, cs285, or scripts")
import argparse

import gymnasium as gym
from infrastructure.buffer import ReplayBuffer
from infrastructure.misc_utils import *
from agents.agent import Agent
import matplotlib.pyplot as plt
from tqdm import tqdm


def training_loop(env_name):
    env = gym.make(get_env(env_name))
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
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
            exp = from_numpy(replay_buffer.sample(batch_size))
            update_info = agent.update(exp["observations"], exp["actions"], exp["rewards"], exp["next_observations"], exp["dones"])
            losses.append(update_info["q_net_loss"])
            q_values.append(update_info["q_value"])
            target_values.append(update_info["target_value"])
        else:
            losses.append(0)
            q_values.append(0)
            target_values.append(0)

    agent.q_net.save(model_path + f"{env_name}_q_net.pt")
    agent.target_net.save(model_path + f"{env_name}_target_net.pt")
    env.close()
    return total_steps, losses, q_values, target_values


def plot_results(total_steps, losses, q_values, target_values):
    plt.plot(range(total_steps), losses, label = "Loss")
    plt.plot(range(total_steps), q_values, label = "Q-value")
    plt.plot(range(total_steps), target_values, label = "Target value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    env_choices = ["cartpole", "ant", "pendulum", "inv_pend", "lander", "hopper"]
    parser.add_argument("--env_name", "-e", required = True, choices = env_choices, help = f"Choices are {env_choices}")
    args = parser.parse_args()
    total_steps, losses, q_values, target_values = training_loop(args.env_name)
    plot_results(total_steps, losses, q_values, target_values)