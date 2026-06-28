import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.resolve()
RESULTS_DIR = os.path.join(BASE_DIR, "results")
#
TABQL_FILE_NAME = "tabular_qlearning_logs.csv"
TABQL_FILE_PATH = os.path.join(RESULTS_DIR, TABQL_FILE_NAME)
DQN_FILE_NAME = "dqn_logs.csv"
DQN_FILE_PATH = os.path.join(RESULTS_DIR, DQN_FILE_NAME)
RINF_FILE_NAME = "reinforce_logs.csv"
RINF_FILE_PATH = os.path.join(RESULTS_DIR, RINF_FILE_NAME)
A2C_FILE_NAME = "a2c_logs.csv"
A2C_FILE_PATH = os.path.join(RESULTS_DIR, A2C_FILE_NAME)
#
PLOT_PATH = os.path.join(RESULTS_DIR, "task4_comparison.png")


def plot_all_agents():
    # 1. Define the configuration for all four agents
    # Make sure these filenames match exactly what your scripts saved!
    agents = {
        "Tabular Q-Learning": {
            "file": TABQL_FILE_PATH,
            "color": "blue",
            "window": 50,  # Larger smoothing window for Tabular
        },
        "DQN": {
            "file": DQN_FILE_PATH,
            "color": "purple",
            "window": 20,
        },
        "REINFORCE": {
            "file": RINF_FILE_PATH,
            "color": "goldenrod",  # Dark yellow for better visibility
            "window": 20,
        },
        "A2C": {
            "file": A2C_FILE_PATH,
            "color": "green",
            "window": 20,
        },
    }

    plt.figure(figsize=(12, 7))

    # 2. Process and plot each agent
    for name, config in agents.items():
        filepath = config["file"]

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping {name}.")
            continue

        print(f"Processing {name}...")
        df = pd.read_csv(filepath)
        seeds = df["Seed"].unique()

        all_steps = []
        all_returns = []

        # Extract data for each seed
        for seed in seeds:
            seed_data = df[df["Seed"] == seed]
            all_steps.append(seed_data["Cumulative_Steps"].values)
            all_returns.append(seed_data["Return"].values)

        # Create a common step grid (0 to the max step this specific agent reached)
        max_step = max([steps.max() for steps in all_steps])
        common_step_grid = np.linspace(0, max_step, 500)

        interpolated_returns = []

        for steps, returns in zip(all_steps, all_returns):
            # Smooth the returns to remove extreme noise
            window = config["window"]
            smoothed_returns = np.convolve(
                returns, np.ones(window) / window, mode="same"
            )

            # Interpolate onto the common grid so we can average the 3 seeds
            interp_ret = np.interp(common_step_grid, steps, smoothed_returns)
            interpolated_returns.append(interp_ret)

        # Calculate Mean and Std across seeds
        interp_array = np.array(interpolated_returns)
        mean_ret = np.mean(interp_array, axis=0)
        std_ret = np.std(interp_array, axis=0, ddof=1)

        # Plot Mean and Shaded Std
        plt.plot(
            common_step_grid, mean_ret, label=name, color=config["color"], linewidth=2
        )
        plt.fill_between(
            common_step_grid,
            mean_ret - std_ret,
            mean_ret + std_ret,
            color=config["color"],
            alpha=0.15,
        )

    # 3. Format the Graph
    plt.title("Empirical Comparison of RL Algorithms on CartPole-v1", fontsize=14)
    plt.xlabel("Total Environment Steps", fontsize=12)
    plt.ylabel("Return (Moving Average)", fontsize=12)

    # Optional: Log scale on the X-axis can be very helpful because A2C/DQN finish in < 50k steps,
    # while Tabular/REINFORCE might take 200k+. If the graph looks too squished, uncomment the line below:
    # plt.xscale('symlog')

    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # 4. Save
    # save_path = "results/final_comparison_plot.png"
    plt.savefig(PLOT_PATH, dpi=300)
    print(f"\nSuccess! Final comparison plot saved to {PLOT_PATH}")


if __name__ == "__main__":
    plot_all_agents()
