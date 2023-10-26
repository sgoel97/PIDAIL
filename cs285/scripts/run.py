import os
import sys
import argparse

import gymnasium as gym
from tqdm import tqdm
import numpy as np

sys.path.append(os.getcwd() + "/cs285/")

from infrastructure.buffer import ReplayBuffer
from infrastructure.misc_utils import *
from infrastructure.state_utils import *
from infrastructure.plotting_utils import *
from agents.agent import Agent


def training_loop(env_name, using_demos, prune):
    # Set up environment, hyperparameters, and data storage
    gym_env_name = get_env(env_name)
    env = gym.make(gym_env_name)
    # TODO: action dim for the agent network really only works with discrete action spaces
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    replay_buffer = ReplayBuffer()
    total_steps = 1000
    non_learning_steps = 50
    batch_size = 32
    losses = []
    q_values = []
    target_values = []

    # If agent is using expert demos to learn instead of learning from scratch,
    # load the expert data into the replay buffer
    if using_demos:
        expert_file_path = f"../experts/expert_data_{gym_env_name}.pkl"
        # TODO: handle the similarity shit better here
        trajectories = create_trajectories(expert_file_path)
        if prune:
            similar_states = get_similar_states(trajectories)
            variances = get_state_collection_variance(
                similar_states, trajectories, agent
            )
            threshold = np.percentile([var[2] for var in variances], 80)
            for traj_idx, state_idx, variance in variances:
                if variance < threshold:
                    state = trajectories[traj_idx].states[state_idx]
                    replay_buffer.insert(
                        state.obs,
                        state.action,
                        state.reward,
                        state.next_obs,
                        state.done,
                    )
        else:
            for trajectory in trajectories:
                for state in trajectory.states:
                    replay_buffer.insert(
                        state.obs,
                        state.action,
                        state.reward,
                        state.next_obs,
                        state.done,
                    )

    # Main training loop
    observation, _ = env.reset()
    for i in tqdm(range(total_steps)):
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        replay_buffer.insert(
            observation, action, reward, next_observation, terminated and not truncated
        )

        if terminated or truncated:
            observation, _ = env.reset()
        else:
            observation = next_observation

        if i > non_learning_steps or using_demos:
            exp = from_numpy(replay_buffer.sample(batch_size))
            update_info = agent.update(
                exp["observations"],
                exp["actions"],
                exp["rewards"],
                exp["next_observations"],
                exp["dones"],
            )
            losses.append(update_info["q_net_loss"])
            q_values.append(update_info["q_value"])
            target_values.append(update_info["target_value"])
        else:
            losses.append(0)
            q_values.append(0)
            target_values.append(0)

    # Save networks
    data_path = save_networks(using_demos, prune, env_name, agent)

    env.close()
    return total_steps, losses, q_values, target_values, data_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    env_choices = ["cartpole", "ant", "pendulum", "inv_pend", "lander", "hopper"]
    parser.add_argument(
        "--env_name",
        "-e",
        required=True,
        choices=env_choices,
        help=f"Choices are {env_choices}",
    )
    parser.add_argument(
        "--demos",
        "-d",
        action="store_true",
        help="Whether the agent is using expert data to learn or learning on its own",
    )
    parser.add_argument(
        "--prune",
        "-p",
        action="store_true",
        help="Whether or not to prune trajectories",
    )
    args = parser.parse_args()
    total_steps, losses, q_values, target_values, data_path = training_loop(
        args.env_name, args.demos, args.prune
    )
    plot_results(total_steps, losses, q_values, target_values, data_path)
