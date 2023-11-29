import matplotlib.pyplot as plt


def plot_results(total_steps, values, names, save_path):
    """
    Plots loss, Q-value, and target value, and saves
    """
    x_range = range(total_steps)

    for value, name in zip(values, names):
        plt.plot(x_range, value, label=name)
    
    plt.legend()
    plt.savefig(save_path / "results.png")
    plt.show()
