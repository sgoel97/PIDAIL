import os
import sys
import argparse
import numpy as np

sys.path.append(os.getcwd() + "/cs285/")

import gymnasium as gym
from datetime import datetime

from huggingface_sb3 import load_from_hub

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MlpPolicy

import tempfile
from imitation.algorithms import bc
from imitation.algorithms.bc import RolloutStatsComputer
from imitation.policies.serialize import load_policy
from imitation.data import rollout
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.algorithms import sqil
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.logger import HierarchicalLogger
from imitation.policies.base import FeedForward32Policy

from infrastructure.misc_utils import *
from infrastructure.imitation_state_utils import *
from infrastructure.plotting_utils import *
from infrastructure.scripting_utils import *
from infrastructure.imitation_agent_utils import *
from infrastructure.custom_data_types import *

discrete_agents = ["dqn", "sqil", "dagger", "bc", "gail", "dqfd"]
continous_agents = ["sac", "td3", "gail", "dagger", "bc"]
cluster_types = ["agglomerative", "kmeans"]
prune_types = ["group", "action", "outcome", "value"]
"""
group: delete entire groups based on which have the highest variance/entropy
action: delete FROM groups based on mode action or mean action
outcome: delete entire groups based on which have the highest variance/entropy amongst future states
value: delete FROM groups based on best-performing action(s)
"""


def training_loop(
    env_name,
    using_demos,
    prune,
    config,
    agent,
    seed,
    num_eval_runs=20,
    init_weight_file=None,
    timestamp=None,
):
    # Set up environment, hyperparameters, and data storage
    total_steps = config["total_steps"]
    gym_env_name = get_env(env_name)

    # Set up environment
    rng = np.random.default_rng(seed=seed)

    # env = Monitor(gym.make(gym_env_name), filename=log_dir + "/train")
    # eval_env = Monitor(gym.make(gym_env_name), filename=log_dir + "/eval")

    env = make_vec_env(
        gym_env_name,
        rng=rng,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )
    eval_env = make_vec_env(
        gym_env_name,
        rng=rng,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )
    dqfd_train_env = gym.make(gym_env_name)
    dqfd_eval_env = gym.make(gym_env_name)

    # Set up defaults
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    agent_name = get_default_agent(agent, discrete, using_demos)
    check_agent_env(agent_name, discrete, discrete_agents, continous_agents)

    # Set up logging
    if timestamp is None:
        timestamp = datetime.now().strftime("%d_%H:%M:%S").replace("/", "_")
    ext = "_pruned" if prune else ""
    log_dir = f"{os.getcwd()}/logs/{env_name}/{agent_name}{ext}_{timestamp}"
    logger = configure(log_dir, ["tensorboard"])
    imitation_logger = HierarchicalLogger(logger)

    # Set up period evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=500,
        verbose=0,
        render=False,
        n_eval_episodes=num_eval_runs,
    )

    eval_returns = []
    episode_lengths = []

    if using_demos:
        expert_file_path = f"{os.getcwd()}/cs285/experts/expert_data_{gym_env_name}.pkl"

        prune_config = config["prune_config"]
        prune_type = prune_config["prune_type"]
        assert (
            prune_type in prune_types
        ), f"prune_type config must be one of {prune_types}"

        if prune and prune_type in ["outcome", "value"]:
            rollouts = create_imitation_trajectories(expert_file_path, custom=True)
            transitions = Trajectory.flatten_trajectories(rollouts)
        else:
            rollouts = create_imitation_trajectories(expert_file_path)
            transitions = rollout.flatten_trajectories_with_rew(rollouts)

        if prune:
            cluster_config = config["cluster_config"]
            cluster_type = cluster_config["cluster_type"]
            assert (
                cluster_type in cluster_types
            ), f"cluster_type config must be one of {cluster_types}"
            if cluster_type == "agglomerative":
                groups = group_transitions_agglomerative(
                    transitions, cluster_config["agglomerative_clustering_kwargs"]
                )
            elif cluster_type == "kmeans":
                groups = group_transitions_kmeans(
                    transitions, cluster_config["kmeans_clustering_kwargs"]
                )

            # print("Before Filtering:\n############################")
            # print_group_stats(groups)

            if prune_type == "group":
                filtered_groups = filter_transition_groups(
                    groups, prune_config["group_filtering_kwargs"]
                )
            elif prune_type == "action":
                filtered_groups = prune_transition_groups(
                    groups, discrete, prune_config["action_filtering_kwargs"]
                )
            elif prune_type == "outcome":
                filtered_groups = filter_groups_by_outcome(
                    rollouts, groups, prune_config["outcome_filtering_kwargs"]
                )
            elif prune_type == "value":
                filtered_groups = prune_groups_by_value(
                    dqfd_train_env,
                    transitions,
                    groups,
                    discrete,
                    prune_config["value_filtering_kwargs"],
                )
            # print("After Filtering:\n############################")
            # print_group_stats(filtered_groups)

            transitions = collate_transitions(filtered_groups)

        # BC
        if agent_name == "bc":
            bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=transitions,
                rng=rng,
                custom_logger=imitation_logger,
            )

            if init_weight_file is not None:
                bc_trainer.policy.load_state_dict(torch.load(init_weight_file))
            else:
                torch.save(
                    bc_trainer.policy.state_dict(), Path(log_dir) / "init_weights.pth"
                )

            def evaluate():
                eval_return, ep_lens = evaluate_policy(
                    bc_trainer.policy,
                    eval_env,
                    n_eval_episodes=num_eval_runs,
                    deterministic=True,
                    return_episode_rewards=True,
                )
                eval_returns.append(eval_return)
                episode_lengths.append(ep_lens)

            bc_trainer.train(
                n_epochs=total_steps // 1000,
                # log_rollouts_venv=eval_env,
                on_epoch_end=evaluate,
                progress_bar=True,
            )
            agent = bc_trainer.policy

        # Dagger
        if agent_name == "dagger":
            try:
                expert = load_policy(
                    "ppo-huggingface",
                    organization="HumanCompatibleAI",
                    env_name=gym_env_name,
                    venv=env,
                )
            except:
                expert = PPO.load(
                    load_from_hub(
                        repo_id=f"sb3/ppo-{gym_env_name}",
                        filename=f"ppo-{gym_env_name}.zip",
                    )
                )

            bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                rng=rng,
                custom_logger=imitation_logger,
            )

            with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
                print(tmpdir)
                dagger_trainer = SimpleDAggerTrainer(
                    venv=env,
                    scratch_dir=tmpdir,
                    expert_policy=expert,
                    bc_trainer=bc_trainer,
                    rng=rng,
                )
                dagger_trainer.train(
                    total_steps, bc_train_kwargs={"progress_bar": True}
                )

            agent = dagger_trainer.policy

        # Soft Q learning
        if agent_name == "sqil":
            sqil_trainer = sqil.SQIL(
                venv=env,
                demonstrations=rollouts,
                policy="MlpPolicy",
                custom_logger=imitation_logger,
            )

            if init_weight_file is not None:
                sqil_trainer.policy.load_state_dict(torch.load(init_weight_file))
            else:
                torch.save(
                    sqil_trainer.policy.state_dict(), Path(log_dir) / "init_weights.pth"
                )

            sqil_trainer.train(
                total_timesteps=total_steps,
                progress_bar=True,
                callback=eval_callback,
            )
            agent = sqil_trainer.policy

        # GAIL
        if agent_name == "gail":
            learner = PPO(
                env=env,
                policy=MlpPolicy,
                batch_size=64,
                ent_coef=0.0,
                learning_rate=0.0004,
                gamma=0.95,
                n_epochs=5,
                seed=42,
            )
            reward_net = BasicRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                normalize_input_layer=RunningNorm,
            )
            gail_trainer = GAIL(
                demonstrations=rollouts,
                demo_batch_size=256,
                gen_replay_buffer_capacity=512,
                n_disc_updates_per_round=8,
                venv=env,
                gen_algo=learner,
                reward_net=reward_net,
                allow_variable_horizon=True,
                custom_logger=imitation_logger,
            )

            def evaluate(_):
                eval_return, ep_lens = evaluate_policy(
                    gail_trainer.policy,
                    eval_env,
                    n_eval_episodes=num_eval_runs,
                    deterministic=True,
                    return_episode_rewards=True,
                )
                eval_returns.append(eval_return)
                episode_lengths.append(ep_lens)

            gail_trainer.train(total_timesteps=total_steps, callback=evaluate)
            agent = gail_trainer.policy

        if agent_name == "dqfd":
            agent = DQfDAgent(dqfd_train_env, pretrain_steps=config["dqfd_pretrain_steps"])
            agent.set_log_dir(log_dir)

            def evaluate():
                eval_return, _, ep_lens = agent.evaluate(
                    config["max_steps_per_traj"],
                    n_eval_episodes=num_eval_runs,
                    eval=False,
                    new_env=dqfd_eval_env,
                )
                eval_returns.append(eval_return)
                episode_lengths.append(ep_lens)

            agent.train(total_steps, transitions, progress_bar=True, callback=evaluate)

    else:
        agent = get_agent(agent_name, env, config)
        if isinstance(agent, DQfDAgent):
            agent.set_log_dir(log_dir)
            agent.train(total_steps, transitions, progress_bar=True, callback=evaluate)
        else:
            agent.set_logger(logger)
            agent.learn(total_steps, callback=eval_callback, progress_bar=True)

    # Evaluate at end
    if isinstance(agent, DQfDAgent):
        eval_return, _, ep_len = agent.evaluate(
            config["max_steps_per_traj"],
            n_eval_episodes=num_eval_runs,
            new_env=dqfd_eval_env,
        )
    else:
        eval_return, ep_len = evaluate_policy(
            agent,
            eval_env,
            n_eval_episodes=num_eval_runs,
            deterministic=True,
            return_episode_rewards=True,
        )
    eval_returns.append(eval_return)
    episode_lengths.append(ep_len)

    if agent_name in ["bc", "gail", "dqfd"]:
        np.savez_compressed(
            log_dir + "/evaluations", results=eval_returns, ep_lengths=episode_lengths
        )
    if prune:
        print("PRUNED RESULTS")
    else:
        print("UNPRUNED RESULTS")
    print("Final avg. eval return:", eval_returns[-1])
    print("Final avg. episode length:", episode_lengths[-1])
    print("Overall avg. eval return:", np.mean(eval_returns))
    print("Overall std. eval return:", np.std(eval_returns))
    print("Overall avg. epsiode length:", np.mean(episode_lengths))
    print("Overall std. episode length:", np.std(episode_lengths))
    print()

    env.close()
    eval_env.close()

    return total_steps, log_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    env_choices = [
        "ant",
        "cartpole",
        "cheetah",
        "hopper",
        "inv_pend",
        "lander",
        "walker",
    ]  # alphabetical order now -jg
    # env_choices = ["cartpole", "ant", "pendulum", "inv_pend", "lander", "hopper"]
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
    parser.add_argument(
        "--seed",
        "-s",
        help=f"random seed for reproducibility",
        default=42,
    )
    parser.add_argument(
        "--eval_runs",
        "-r",
        help="number of eval runs",
        default=20,
    )

    args = parser.parse_args()

    if not args.demos and args.prune:
        raise NotImplementedError("Can't use prune without expert demos")

    config = make_config(f"{os.getcwd()}/cs285/configs/{args.env_name}.yaml")

    total_steps, log_dir = training_loop(
        args.env_name,
        args.demos,
        args.prune,
        config,
        agent=args.agent,
        seed=args.seed,
        num_eval_runs=int(args.eval_runs),
    )

    plot_npz(log_dir + "/evaluations.npz", log_dir, show=args.graph)
