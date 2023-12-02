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
from infrastructure.eval_utils import *
from infrastructure.scripting_utils import *
from agents.agent import Agent
from agents.cagent import CAgent


total_steps = 10000
non_learning_steps = 100
batch_size = 100


def training_loop(env_name, using_demos, prune, config):
    # Set up environment, hyperparameters, and data storage
    gym_env_name = get_env(env_name)
    env = gym.make(gym_env_name)
    eval_env = gym.make(gym_env_name)

    # Set up agents
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    obs_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    if discrete:
        agent = Agent(obs_dim, ac_dim, **config)
    else:
        agent = CAgent(obs_dim, ac_dim, **config)

    # Set up replay buffer
    replay_buffer = ReplayBuffer()

    # Set up logging
    losses, q_values, target_values = [], [], []
    if not discrete:
        actor_losses = []

    # If agent is using expert demos to learn instead of learning from scratch,
    # load the expert data into the replay buffer
    if using_demos:
        expert_file_path = f"{os.getcwd()}/cs285/experts/expert_data_{gym_env_name}.pkl"

        trajectories = create_trajectories(expert_file_path)

        if prune:
            transition_groups = get_similar_transitions(trajectories)
            avg_group_size = np.mean(list(map(len, transition_groups)))
            print(f"Number of transition groups: {len(transition_groups)}")
            print(f"Average group size: {avg_group_size}")

            filtered_transition_groups = filter_transition_groups(
                transition_groups, size_treshold=8, measure_cutoff=80
            )
            avg_group_size = np.mean(list(map(len, filtered_transition_groups)))
            print(f"Number of filtered groups: {len(filtered_transition_groups)}")
            print(f"Average filtered group size: {avg_group_size}")

            for group in filtered_transition_groups:
                for transition in group:
                    replay_buffer.insert(
                        transition.obs,
                        transition.action,
                        transition.reward,
                        transition.next_obs,
                        transition.done,
                    )
        else:
            for trajectory in trajectories:
                for transition in trajectory.transitions:
                    replay_buffer.insert(
                        transition.obs,
                        transition.action,
                        transition.reward,
                        transition.next_obs,
                        transition.done,
                    )

        print(f"Replay buffer size: {len(replay_buffer)}")

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

            if not discrete:
                actor_losses.append(update_info["actor_loss"])
        else:
            losses.append(0)
            q_values.append(0)
            target_values.append(0)

            if not discrete:
                actor_losses.append(0)

    # Evaluate at end
    trajectories = sample_n_trajectories(
        eval_env, policy=agent, ntraj=10, max_length=10000
    )
    returns = [t["episode_statistics"]["r"] for t in trajectories]
    ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

    print("eval return:", np.mean(returns))
    print("episode length:", np.mean(ep_lens))

    # Save networks
    data_path = save_networks(using_demos, prune, env_name, agent)

    env.close()

    results = {"loss": losses, "q_values": q_values, "target_values": target_values}
    if not discrete:
        results["actor_loss"] = actor_losses

    return total_steps, results, data_path


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
    parser.add_argument(
        "--graph",
        "-g",
        action="store_true",
        help="Whether or not to graph results",
    )

    args = parser.parse_args()

    config = make_config(f"{os.getcwd()}/cs285/configs/{args.env_name}.yaml")

    if not args.demos and args.prune:
        raise NotImplementedError("Can't use prune without expert demos")

    total_steps, results, data_path = training_loop(
        args.env_name, args.demos, args.prune, config
    )

    if args.graph:
        plot_results(total_steps, results.values(), results.keys(), data_path)
