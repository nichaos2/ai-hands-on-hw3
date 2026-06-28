import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agents.a2c import A2CAgent
from utils.seeding import set_global_seed


def run_experiment():
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    seeds = [42, 1337, 2024]
    max_episodes = 1000

    log_data = []
    all_runs_steps = []
    all_runs_returns = []

    experiment_start_time = time.time()

    for seed in seeds:
        print(f"Training A2C (Seed {seed})...")
        env = gym.make("CartPole-v1")
        set_global_seed(seed, env)

        # Hyperparameters strictly follow the requested ranges
        agent = A2CAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            lr=1e-3,
            gamma=0.99,
            entropy_beta=0.01,  # beta in [0.001, 0.05]
            critic_coef=0.5,  # c_v in [0.25, 1.0]
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
                # Retrieve all required computational graph components for the update
                action_tensor, action_int, value, log_prob, entropy = (
                    agent.get_action_and_value(state)
                )

                next_state, reward, terminated, truncated, _ = env.step(action_int)
                done = terminated or truncated

                # Perform the 1-step online TD update immediately
                agent.update(
                    state,
                    action_tensor,
                    reward,
                    next_state,
                    done,
                    log_prob,
                    entropy,
                    value,
                )

                state = next_state
                total_reward += reward
                cumulative_steps += 1

            seed_steps.append(cumulative_steps)
            seed_returns.append(total_reward)

            log_data.append(
                {
                    "Algorithm": "A2C",
                    "Seed": seed,
                    "Episode": episode,
                    "Cumulative_Steps": cumulative_steps,
                    "Return": total_reward,
                }
            )

        all_runs_steps.append(seed_steps)
        all_runs_returns.append(seed_returns)

        # Save final weights
        torch.save(agent.network.state_dict(), f"checkpoints/a2c_seed_{seed}.pt")
        env.close()

    # --- Data Aggregation and Output ---
    experiment_end_time = time.time()
    total_time_min = (experiment_end_time - experiment_start_time) / 60.0

    # Save training logs for the final plot
    pd.DataFrame(log_data).to_csv("results/a2c_logs.csv", index=False)

    # Calculate Final Table Metrics (Last 100 episodes)
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
        "Algorithm": ["A2C"],
        "Steps to solve": [steps_to_solve],
        "Final return (mean)": [round(final_return_mean, 2)],
        "Std": [round(final_return_std, 2)],
        "Wall-clock (min)": [round(total_time_min, 2)],
    }
    pd.DataFrame(summary_data).to_csv("results/a2c_comparing_table.csv", index=False)
    print("Saved A2C logs and comparison metrics to results directory.")

    # --- Plotting (Interpolated over Environment Steps) ---
    print("\nGenerating A2C Plot...")
    plt.figure(figsize=(10, 6))

    # 1. Create a common step grid for interpolation
    max_step = max([max(steps) for steps in all_runs_steps])
    common_step_grid = np.linspace(0, max_step, 500)

    interpolated_returns = []

    # 2. Interpolate each seed's data onto the common grid
    for steps, returns in zip(all_runs_steps, all_runs_returns):
        # Smooth with a 20-episode moving average
        smoothed_returns = np.convolve(returns, np.ones(20) / 20, mode="same")
        interp_ret = np.interp(common_step_grid, steps, smoothed_returns)
        interpolated_returns.append(interp_ret)

    # 3. Calculate Mean and Std
    interp_array = np.array(interpolated_returns)
    mean_ret = np.mean(interp_array, axis=0)
    std_ret = np.std(interp_array, axis=0, ddof=1)

    # 4. Plot (Using green to distinguish from DQN and REINFORCE)
    plt.plot(common_step_grid, mean_ret, label="A2C", color="green")
    plt.fill_between(
        common_step_grid,
        mean_ret - std_ret,
        mean_ret + std_ret,
        color="green",
        alpha=0.2,
    )

    plt.title("A2C on CartPole-v1: Return vs Total Steps")
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Return (20-Episode Moving Average)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 5. Save the plot
    plt.savefig("results/task3_a2c.png")
    print("Saved plot to results/task3_a2c.png")


if __name__ == "__main__":
    run_experiment()
