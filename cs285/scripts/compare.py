import os
import argparse
from pathlib import Path
from datetime import datetime

from run import training_loop
from infrastructure.scripting_utils import *
from infrastructure.plotting_utils import *

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
        "--graph",
        "-g",
        action="store_true",
        help="Whether or not to graph results",
    )

    args = parser.parse_args()

    config = make_config(f"{os.getcwd()}/cs285/configs/{args.env_name}.yaml")

    total_steps, unpruned_log_dir = training_loop(
        args.env_name, using_demos=True, prune=False, config=config, agent="bc", seed=42
    )

    total_steps, pruned_log_dir = training_loop(
        args.env_name, using_demos=True, prune=True, config=config, agent="bc", seed=42
    )

    rollout_stats = [
        "rollout/return_max",
        "rollout/return_mean",
        "rollout/return_min",
        "rollout/return_std",
    ]

    unpruned_stats = parse_tensorboard(
        unpruned_log_dir,
        rollout_stats,
    )

    pruned_stats = parse_tensorboard(
        pruned_log_dir,
        rollout_stats,
    )

    timestamp = datetime.now().strftime("%d_%H:%M:%S").replace("/", "_")
    log_dir = f"{os.getcwd()}/logs/{args.env_name}/bc_comparison_{timestamp}"

    unpruned_dfs = list(map(lambda x: x.value, unpruned_stats.values()))
    pruned_dfs = list(map(lambda x: x.value, pruned_stats.values()))

    unpruned_keys = ["unpruned_" + k for k in list(unpruned_stats.keys())]
    pruned_keys = ["pruned_" + k for k in list(pruned_stats.keys())]

    for i, file_ext in enumerate(["max", "mean", "min", "std"]):
        plot_comparison_results(
            unpruned_dfs[i : i + 1],
            pruned_dfs[i : i + 1],
            unpruned_keys[i : i + 1],
            pruned_keys[i : i + 1],
            log_dir,
            show=False,
            file_ext=file_ext,
        )

    if args.graph:
        plot_comparison_results(
            unpruned_dfs, pruned_dfs, unpruned_keys, pruned_keys, log_dir, show=True
        )
