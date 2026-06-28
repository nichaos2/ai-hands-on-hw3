import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agents.reinforce import REINFORCEAgent
from utils.seeding import set_global_seed


def run_experiment():
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    seeds = [42, 1337, 2024]
    max_episodes = 1000  # REINFORCE typically needs more episodes than DQN

    all_runs_steps = []
    all_runs_returns = []
    log_data = []

    experiment_start_time = time.time()

    for seed in seeds:
        print(f"Training REINFORCE (Seed {seed})...")
        env = gym.make("CartPole-v1")
        set_global_seed(seed, env)

        agent = REINFORCEAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            lr=1e-3,
            gamma=0.99,
            device="cpu",
        )

        seed_steps = []
        seed_returns = []
        cumulative_steps = 0

        for episode in range(1, max_episodes + 1):
            state, _ = env.reset(seed=seed if episode == 1 else None)
            total_reward = 0
            done = False

            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_reward(reward)

                state = next_state
                total_reward += reward
                cumulative_steps += 1

            # Perform the gradient update at the end of the episode
            agent.update()

            seed_steps.append(cumulative_steps)
            seed_returns.append(total_reward)

            log_data.append(
                {
                    "Algorithm": "REINFORCE",
                    "Seed": seed,
                    "Episode": episode,
                    "Cumulative_Steps": cumulative_steps,
                    "Return": total_reward,
                }
            )

        all_runs_steps.append(seed_steps)
        all_runs_returns.append(seed_returns)

        # Save final weights for this seed
        torch.save(
            agent.policy_network.state_dict(), f"checkpoints/reinforce_seed_{seed}.pt"
        )
        env.close()

    # --- Metrics & Logging ---
    experiment_end_time = time.time()
    total_time_min = (experiment_end_time - experiment_start_time) / 60.0

    pd.DataFrame(log_data).to_csv("results/reinforce_logs.csv", index=False)

    # Calculate Table Metrics (Last 100 episodes)
    final_returns = [np.mean(returns[-100:]) for returns in all_runs_returns]
    final_return_mean = np.mean(final_returns)
    final_return_std = np.std(final_returns, ddof=1)

    # Calculate steps to solve
    steps_to_solve = "Did not solve"
    for s_steps, s_returns in zip(all_runs_steps, all_runs_returns):
        s_returns_array = np.array(s_returns)
        if len(s_returns_array) >= 100:
            moving_avg = np.convolve(s_returns_array, np.ones(100) / 100, mode="valid")
            solve_idx = np.where(moving_avg >= 475.0)[0]
            if len(solve_idx) > 0:
                solve_ep = solve_idx[0] + 99
                steps_to_solve = s_steps[solve_ep]
                break

    summary_data = {
        "Algorithm": ["REINFORCE"],
        "Steps to solve": [steps_to_solve],
        "Final return (mean)": [round(final_return_mean, 2)],
        "Std": [round(final_return_std, 2)],
        "Wall-clock (min)": [round(total_time_min, 2)],
    }
    pd.DataFrame(summary_data).to_csv(
        "results/reinforce_comparing_table.csv", index=False
    )
    print("Saved REINFORCE logs and comparison metrics.")

    # --- Plotting (Interpolated over Environment Steps) ---
    plt.figure(figsize=(10, 6))

    max_step = max([max(steps) for steps in all_runs_steps])
    common_step_grid = np.linspace(0, max_step, 500)

    interpolated_returns = []
    for steps, returns in zip(all_runs_steps, all_runs_returns):
        smoothed_returns = np.convolve(returns, np.ones(20) / 20, mode="same")
        interp_ret = np.interp(common_step_grid, steps, smoothed_returns)
        interpolated_returns.append(interp_ret)

    interp_array = np.array(interpolated_returns)
    mean_ret = np.mean(interp_array, axis=0)
    std_ret = np.std(interp_array, axis=0, ddof=1)

    plt.plot(common_step_grid, mean_ret, label="REINFORCE", color="orange")
    plt.fill_between(
        common_step_grid,
        mean_ret - std_ret,
        mean_ret + std_ret,
        color="orange",
        alpha=0.2,
    )

    plt.title("REINFORCE on CartPole-v1: Return vs Total Steps")
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Return (20-Episode Moving Average)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/task3_reinforce.png")
    print("Saved plot to results/task3_reinforce.png")


if __name__ == "__main__":
    run_experiment()
