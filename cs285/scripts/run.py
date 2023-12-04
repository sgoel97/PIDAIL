import os
import sys
import argparse

sys.path.append(os.getcwd() + "/cs285/")

import gymnasium as gym
from datetime import datetime

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from infrastructure.misc_utils import *
from infrastructure.state_utils import *
from infrastructure.plotting_utils import *
from infrastructure.scripting_utils import *
from infrastructure.agent_utils import *

discrete_agents = ["dqn"]
continous_agents = ["sac", "td3"]


def training_loop(env_name, using_demos, prune, config, agent=None):
    # Set up environment, hyperparameters, and data storage
    total_steps = config["total_steps"]
    gym_env_name = get_env(env_name)

    # Set up defaults
    discrete = isinstance(gym.make(gym_env_name).action_space, gym.spaces.Discrete)
    agent_name = get_default_agent(agent, discrete)

    # Set up logging
    timestamp = datetime.now().strftime("%d_%H:%M:%S").replace("/", "_")
    log_dir = f"{os.getcwd()}/logs/{env_name}/{agent_name}_{timestamp}"
    logger = configure(log_dir, ["tensorboard"])

    # Set up environment
    env = Monitor(gym.make(gym_env_name), filename=log_dir + "/train")
    eval_env = Monitor(gym.make(gym_env_name), filename=log_dir + "/eval")

    # Set up agents
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    agent_name = get_default_agent(agent, discrete)
    check_agent_env(agent_name, discrete, discrete_agents, continous_agents)
    agent = get_agent(agent_name, env, config)
    agent.set_logger(logger)

    # Set up period evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=500,
        verbose=0,
        render=False,
    )

    # Set up replay buffer
    # replay_buffer = ReplayBuffer()

    # If agent is using expert demos to learn instead of learning from scratch,
    # load the expert data into the replay buffer
    # if using_demos:
    #     expert_file_path = f"{os.getcwd()}/cs285/experts/expert_data_{gym_env_name}.pkl"

    #     trajectories = create_trajectories(expert_file_path)

    #     if prune:
    #         transition_groups = get_similar_transitions(trajectories)
    #         avg_group_size = np.mean(list(map(len, transition_groups)))
    #         print(f"Number of transition groups: {len(transition_groups)}")
    #         print(f"Average group size: {avg_group_size}")

    #         filtered_transition_groups = filter_transition_groups(
    #             transition_groups, size_treshold=8, measure_cutoff=80
    #         )
    #         avg_group_size = np.mean(list(map(len, filtered_transition_groups)))
    #         print(f"Number of filtered groups: {len(filtered_transition_groups)}")
    #         print(f"Average filtered group size: {avg_group_size}")

    #         for group in filtered_transition_groups:
    #             for transition in group:
    #                 replay_buffer.insert(
    #                     transition.obs,
    #                     transition.action,
    #                     transition.reward,
    #                     transition.next_obs,
    #                     transition.done,
    #                 )
    #     else:
    #         for trajectory in trajectories:
    #             for transition in trajectory.transitions:
    #                 replay_buffer.insert(
    #                     transition.obs,
    #                     transition.action,
    #                     transition.reward,
    #                     transition.next_obs,
    #                     transition.done,
    #                 )

    #     print(f"Replay buffer size: {len(replay_buffer)}")

    # Main training loop
    agent.learn(total_timesteps=total_steps, callback=eval_callback, progress_bar=True)

    # Evaluate at end
    avg_eval_return, std_eval_return = evaluate_policy(
        agent, eval_env, n_eval_episodes=10
    )

    print("These arent that accurate:\n#################")
    print("avg. eval return:", avg_eval_return)
    print("std. eval return:", std_eval_return)

    env.close()
    eval_env.close()

    return total_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    env_choices = ["cartpole", "ant", "pendulum", "inv_pend", "lander", "hopper"]
    agent_choices = discrete_agents + continous_agents

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
    parser.add_argument(
        "--agent",
        "-a",
        choices=agent_choices,
        help=f"Choices are {agent_choices}",
        default=None,
    )

    args = parser.parse_args()

    if not args.demos and args.prune:
        raise NotImplementedError("Can't use prune without expert demos")

    config = make_config(f"{os.getcwd()}/cs285/configs/{args.env_name}.yaml")

    total_steps = training_loop(
        args.env_name, args.demos, args.prune, config, agent=args.agent
    )

    # if args.graph:
    #     plot_results(total_steps, results.values(), results.keys(), data_path)
