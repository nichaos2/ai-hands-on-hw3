import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents.tabular_q import TabularQAgent
from utils.seeding import set_global_seed


def create_bins():
    """Creates uniform bins for the 4 CartPole variables."""
    bins = [
        np.linspace(-2.4, 2.4, 5),
        np.linspace(-3.0, 3.0, 5),
        np.linspace(-0.209, 0.209, 5),
        np.linspace(-3.0, 3.0, 5),
    ]
    return bins


def discretize_state(state: np.ndarray, bins: list) -> tuple:
    """Maps continuous state variables to discrete integer bins."""
    return tuple(np.digitize(s, b) for s, b in zip(state, bins))


def run_experiment():
    # 1. Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # env = gym.make("CartPole-v1")
    seeds = [42, 1337, 2024]
    n_episodes = 2000

    bins = create_bins()
    state_dimensions = (6, 6, 6, 6)

    all_returns = []
    log_data = []  # List to hold our CSV row data

    # Track overall wall-clock time
    experiment_start_time = time.time()

    for seed in seeds:
        print(f"Training Tabular Q-Learning (Seed {seed})...")
        env = gym.make("CartPole-v1")
        set_global_seed(seed, env)

        agent = TabularQAgent(
            state_bins=state_dimensions,
            n_actions=env.action_space.n,
            lr=0.1,
            gamma=0.99,
        )

        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay_steps = 1500

        seed_returns = []
        cumulative_steps = 0

        for episode in range(1, n_episodes + 1):
            state, _ = env.reset(seed=seed if episode == 1 else None)
            state_idx = discretize_state(state, bins)

            epsilon = max(epsilon_end, epsilon_start - (episode / epsilon_decay_steps))

            total_reward = 0
            done = False

            while not done:
                action = agent.get_action(state_idx, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_state_idx = discretize_state(next_state, bins)
                agent.update(state_idx, action, reward, next_state_idx, done)

                state_idx = next_state_idx
                total_reward += reward
                cumulative_steps += 1

            seed_returns.append(total_reward)

            # Log the required metrics per episode
            log_data.append(
                {
                    "Algorithm": "Tabular Q-Learning",
                    "Seed": seed,
                    "Episode": episode,
                    "Cumulative_Steps": cumulative_steps,
                    "Return": total_reward,
                }
            )
            # end of episodes
        env.close()  # end of seed
        all_returns.append(seed_returns)

        # Save the final Q-table for this seed
        np.save(f"checkpoints/tabular_q_seed_{seed}.npy", agent.q_table)

    experiment_end_time = time.time()
    total_time_min = (experiment_end_time - experiment_start_time) / 60.0
    print(f"Total Wall-clock time: {total_time_min:.2f} minutes")

    env.close()

    # --- 2. Save CSV Logs ---
    df_logs = pd.DataFrame(log_data)
    csv_path = "results/tabular_qlearning_logs.csv"
    df_logs.to_csv(csv_path, index=False)
    print(f"Saved training logs to {csv_path}")

    # --- 3. Plotting (Episodes vs Return) ---
    returns_array = np.array(all_returns)
    mean_returns = np.mean(returns_array, axis=0)
    std_returns = np.std(returns_array, axis=0, ddof=1)

    window = 50
    mean_smoothed = np.convolve(mean_returns, np.ones(window) / window, mode="valid")
    std_smoothed = np.convolve(std_returns, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 6))
    plt.plot(mean_smoothed, label="Tabular Q-Learning")
    plt.fill_between(
        range(len(mean_smoothed)),
        mean_smoothed - std_smoothed,
        mean_smoothed + std_smoothed,
        alpha=0.2,
    )
    plt.title("Tabular Q-Learning on CartPole-v1 (Moving Average = 50)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/task1_tabular_qlearning.png")
    print("Saved plot to results/task1_tabular_qlearning.png")

    last_100_episodes = returns_array[:, -100:]

    # Get the average score for each seed over its final 100 episodes
    seed_final_scores = np.mean(last_100_episodes, axis=1)

    # Calculate the overall mean and std across the 3 seeds
    final_return_mean = np.mean(seed_final_scores)
    final_return_std = np.std(seed_final_scores, ddof=1)

    # Calculate steps to solve (first time the 100-episode moving average hits 475)
    steps_to_solve = "Did not solve"
    # We can check this by looking at our mean_smoothed array
    solve_indices = np.where(mean_smoothed >= 475.0)[0]
    if len(solve_indices) > 0:
        # Get the episode where it was solved
        solve_episode = solve_indices[0] + window
        # Find cumulative steps at this episode for a baseline seed (e.g., first seed)
        # We approximate by filtering the log data
        df_logs = pd.DataFrame(log_data)
        solve_step = df_logs[
            (df_logs["Seed"] == seeds[0]) & (df_logs["Episode"] == solve_episode)
        ]["Cumulative_Steps"].values[0]
        steps_to_solve = int(solve_step)

    # Create the dataframe for the summary table
    summary_data = {
        "Algorithm": ["Tabular Q-learning"],
        "Steps to solve": [steps_to_solve],
        "Final return (mean)": [round(final_return_mean, 2)],
        "Std": [round(final_return_std, 2)],
        "Wall-clock (min)": [round(total_time_min, 2)],
    }

    df_summary = pd.DataFrame(summary_data)
    summary_csv_path = "results/tabular_q_learning_comparing_table.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved final comparison metrics to {summary_csv_path}")


if __name__ == "__main__":
    run_experiment()
