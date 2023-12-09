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
    agent_choices = ["bc", "sqil", "gail"]

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

    parser.add_argument(
        "--agent",
        "-a",
        choices=agent_choices,
        default="bc",
        help="Which agent to train",
    )

    args = parser.parse_args()

    config = make_config(f"{os.getcwd()}/cs285/configs/{args.env_name}.yaml")

    total_steps, unpruned_log_dir = training_loop(
        args.env_name,
        using_demos=True,
        prune=False,
        config=config,
        agent=args.agent,
        seed=42,
    )

    unpruned_init_weight_file = Path(unpruned_log_dir) / "init_weights.pth"

    total_steps, pruned_log_dir = training_loop(
        args.env_name,
        using_demos=True,
        prune=True,
        config=config,
        agent=args.agent,
        seed=42,
        init_weight_file=unpruned_init_weight_file,
    )

    timestamp = datetime.now().strftime("%d_%H:%M:%S").replace("/", "_")
    log_dir = f"{os.getcwd()}/logs/{args.env_name}/{args.agent}_comparison_{timestamp}"
    plot_compared_npzs(
        unpruned_log_dir + "/evaluations.npz",
        "unpruned",
        pruned_log_dir + "/evaluations.npz",
        "pruned",
        log_dir,
        show=args.graph,
    )
