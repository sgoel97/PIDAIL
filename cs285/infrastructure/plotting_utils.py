import matplotlib.pyplot as plt
from pathlib import Path


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
    values1, values2, names1, names2, save_path, show=False, file_ext=""
):
    """
    Plots loss, Q-value, and target value, and saves
    """
    save_path = Path(save_path)

    for value, name in zip(values1, names1):
        x_range = range(len(value))
        plt.plot(x_range, value, label=name)

    for value, name in zip(values2, names2):
        x_range = range(len(value))
        plt.plot(x_range, value, label=name)

    save_path.mkdir(exist_ok=True, parents=True)
    plt.legend()

    file_ext = file_ext + "_" if file_ext != "" else file_ext
    plt.savefig(save_path / f"{file_ext}results.png")
    if show:
        plt.show()
    else:
        plt.clf()
