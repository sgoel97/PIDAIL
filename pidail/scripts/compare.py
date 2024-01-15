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
    agent_choices = ["bc", "sqil", "gail", "dqfd"]

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
        help="Whether or not to show graph of results",
    )

    parser.add_argument(
        "--agent",
        "-a",
        choices=agent_choices,
        default="bc",
        help="Which agent to train",
    )

    parser.add_argument(
        "--seed",
        "-s",
        default=42,
        help="Which random seed to use",
    )

    parser.add_argument(
        "--eval_runs",
        "-r",
        help="number of eval runs",
        default=20,
    )

    args = parser.parse_args()

    config = make_config(f"{os.getcwd()}/pidail/configs/{args.env_name}.yaml")
    timestamp = datetime.now().strftime("%d_%H:%M:%S").replace("/", "_")

    total_steps, pruned_log_dir = training_loop(
        args.env_name,
        using_demos=True,
        prune=True,
        config=config,
        agent=args.agent,
        seed=args.seed,
        timestamp=timestamp,
        num_eval_runs=int(args.eval_runs),
    )
    plot_npz(pruned_log_dir + "/evaluations.npz", pruned_log_dir)

    pruned_init_weight_file = Path(pruned_log_dir) / "init_weights.pth"

    total_steps, unpruned_log_dir = training_loop(
        args.env_name,
        using_demos=True,
        prune=False,
        config=config,
        agent=args.agent,
        seed=args.seed,
        init_weight_file=pruned_init_weight_file,
        timestamp=timestamp,
        num_eval_runs=int(args.eval_runs),
    )
    plot_npz(unpruned_log_dir + "/evaluations.npz", unpruned_log_dir)

    log_dir = f"{os.getcwd()}/logs/{args.env_name}/{args.agent}_comparison_{timestamp}"

    plot_compared_npzs(
        unpruned_log_dir + "/evaluations.npz",
        "unpruned",
        pruned_log_dir + "/evaluations.npz",
        "pruned",
        log_dir,
        show=args.graph,
    )
