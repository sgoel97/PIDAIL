import matplotlib.pyplot as plt


def plot_results(total_steps, losses, q_values, target_values, save_path):
    """
    Plots loss, Q-value, and target value, and saves
    """
    x_range = range(total_steps)

    plt.plot(x_range, losses, label="Loss")
    plt.plot(x_range, q_values, label="Q-value")
    plt.plot(x_range, target_values, label="Target value")
    plt.legend()
    plt.savefig(save_path / "results.png")
    plt.show()
