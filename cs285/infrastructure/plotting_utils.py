import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd


def plot_results(total_steps, values, names, save_path):
    """
    Plots loss, Q-value, and target value, and saves
    """
    save_path = Path(save_path)
    x_range = range(len(values[0]))

    for value, name in zip(values, names):
        plt.plot(x_range, value, label=name)

    plt.legend()
    plt.savefig(save_path / "results.png")
    plt.show()


def plot_comparison_results(
    max1,
    min1,
    mean1,
    name1,
    max2,
    min2,
    mean2,
    name2,
    save_path,
    show=False,
    file_ext="",
):
    """
    Plots loss, Q-value, and target value, and saves
    """
    save_path = Path(save_path)

    plt.plot(mean1, label=name1, color="blue")
    plt.fill_between(range(len(mean1)), min1, mean1, color="blue", alpha=0.15)
    plt.fill_between(range(len(mean1)), mean1, max1, color="blue", alpha=0.15)

    plt.plot(mean2, label=name2, color="orange")
    plt.fill_between(range(len(mean2)), min2, mean2, color="orange", alpha=0.3)
    plt.fill_between(range(len(mean2)), mean2, max2, color="orange", alpha=0.3)

    save_path.mkdir(exist_ok=True, parents=True)
    plt.legend()

    file_ext = file_ext + "_" if file_ext != "" else file_ext
    plt.savefig(save_path / f"{file_ext}results.png")
    if show:
        plt.show()
    else:
        plt.clf()


def parse_npz(npz_file):
    data = np.load(npz_file)
    eval_returns = data["results"]
    episode_lengths = data["ep_lengths"]

    eval_means = np.mean(eval_returns, axis=1)
    eval_stds = np.std(eval_returns, axis=1)

    episode_means = np.mean(episode_lengths, axis=1)
    episode_stds = np.std(episode_lengths, axis=1)

    return eval_means, eval_stds, episode_means, episode_stds


def plot_npz(npz_file, save_path, show=False):
    save_path = Path(save_path)
    eval_means, eval_stds, episode_means, episode_stds = parse_npz(npz_file)

    plt.plot(eval_means, label="eval return")
    plt.fill_between(
        range(len(eval_means)),
        eval_means - eval_stds,
        eval_means + eval_stds,
        alpha=0.15,
    )

    plt.legend()
    plt.savefig(save_path / "results.png")
    if show:
        plt.show()
    plt.clf()


def plot_compared_npzs(npz_file1, name1, npz_file2, name2, save_path, show=False):
    save_path = Path(save_path)

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    eval_means1, eval_stds1, episode_means1, episode_stds1 = parse_npz(npz_file1)
    eval_means2, eval_stds2, episode_means2, episode_stds2 = parse_npz(npz_file2)

    plt.plot(eval_means1, label=name1 + " eval return", color="blue")
    plt.fill_between(
        range(len(eval_means1)),
        eval_means1 - eval_stds1,
        eval_means1 + eval_stds1,
        alpha=0.15,
    )

    plt.plot(eval_means2, label=name2 + " eval return", color="orange")
    plt.fill_between(
        range(len(eval_means2)),
        eval_means2 - eval_stds2,
        eval_means2 + eval_stds2,
        alpha=0.25,
    )

    plt.legend()
    plt.savefig(save_path / "results.png")
    if show:
        plt.show()


def parse_gail(log_dir):
    gen_csv = Path(log_dir) / "raw/gen/progress.csv"
    df = pd.read_csv(gen_csv)
    eval_return = df["raw/gen/rollout/ep_rew_mean"]
    episode_lenghts = df["raw/gen/rollout/ep_len_mean"]
    return eval_return, episode_lenghts


def plot_gail_comparison(log_dir1, name1, log_dir2, name2, save_path, show=False):
    save_path = Path(save_path)

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    eval_return1, episode_lengths1 = parse_gail(log_dir1)
    eval_return2, episode_lengths2 = parse_gail(log_dir2)

    plt.plot(eval_return1, label=name1 + " eval return", color="blue")
    plt.plot(eval_return2, label=name2 + " eval return", color="orange")

    plt.legend()
    plt.savefig(save_path / "results.png")
    if show:
        plt.show()
