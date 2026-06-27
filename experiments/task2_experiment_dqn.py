import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agents.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.seeding import set_global_seed


def run_experiment():
    # 1. Setup Directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    env = gym.make("CartPole-v1")
    seeds = [42, 1337, 2024]
    max_episodes = 350
    batch_size = 64

    # Epsilon decay parameters (based on steps, not episodes)
    epsilon_start = 1.0
    epsilon_min = 0.05
    epsilon_decay_steps = 10000  # Decay over the first 10k environment steps

    # Ablation Configuration
    variants = {
        "DQN-full": {"buffer_size": 10000, "target_freq": 200, "warmup": 1000},
        "DQN-noReplay": {"buffer_size": batch_size, "target_freq": 200, "warmup": 0},
        "DQN-noTarget": {"buffer_size": 10000, "target_freq": 1, "warmup": 1000},
    }

    # Dictionary to hold plotting data
    variant_results = {}
    epsilon_history = []  # For task2_dqn_epsilon.png

    for variant_name, config in variants.items():
        print(f"\n{'=' * 40}\nStarting Variant: {variant_name}\n{'=' * 40}")
        variant_results[variant_name] = []

        # Logging structures specifically for DQN-full
        full_log_data = []
        experiment_start_time = time.time()

        for seed in seeds:
            print(f"  Training Seed {seed}...")
            env = gym.make("CartPole-v1")
            set_global_seed(seed, env)

            agent = DQNAgent(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                lr=1e-3,
                gamma=0.99,
                target_update_freq=config["target_freq"],
                device="cpu",
            )

            replay_buffer = ReplayBuffer(
                state_dim=env.observation_space.shape[0],
                max_size=config["buffer_size"],
                device="cpu",
            )

            # --- Warm-up Phase ---
            if config["warmup"] > 0:
                state, _ = env.reset(seed=seed)
                while replay_buffer.size < config["warmup"]:
                    action = env.action_space.sample()
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state if not done else env.reset()[0]

            # --- Main Training Loop ---
            seed_steps = []
            seed_returns = []
            cumulative_steps = 0

            for episode in range(1, max_episodes + 1):
                state, _ = env.reset(seed=seed if episode == 1 else None)
                total_reward = 0
                done = False

                while not done:
                    # Calculate epsilon
                    epsilon = max(
                        epsilon_min,
                        epsilon_start
                        - (cumulative_steps / epsilon_decay_steps)
                        * (epsilon_start - epsilon_min),
                    )

                    # Track epsilon for plotting (only need to do this once, e.g., on first seed of full variant)
                    if variant_name == "DQN-full" and seed == seeds[0]:
                        epsilon_history.append(epsilon)

                    action = agent.get_action(state, epsilon)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    replay_buffer.add(state, action, reward, next_state, done)
                    agent.update(replay_buffer, batch_size=batch_size)

                    state = next_state
                    total_reward += reward
                    cumulative_steps += 1

                seed_steps.append(cumulative_steps)
                seed_returns.append(total_reward)

                # Deliverables: Log only for DQN-full
                if variant_name == "DQN-full":
                    full_log_data.append(
                        {
                            "Algorithm": "DQN",
                            "Seed": seed,
                            "Episode": episode,
                            "Cumulative_Steps": cumulative_steps,
                            "Return": total_reward,
                        }
                    )

            variant_results[variant_name].append((seed_steps, seed_returns))

            # Deliverables: Save Checkpoints only for DQN-full
            if variant_name == "DQN-full":
                torch.save(
                    agent.q_network.state_dict(), f"checkpoints/dqn_full_seed_{seed}.pt"
                )

            env.close()

        # Deliverables: Compile logs and summary table for DQN-full
        if variant_name == "DQN-full":
            experiment_end_time = time.time()
            total_time_min = (experiment_end_time - experiment_start_time) / 60.0

            # Save raw logs
            df_logs = pd.DataFrame(full_log_data)
            df_logs.to_csv("results/dqn_logs.csv", index=False)

            # Calculate metrics for the comparison table (last 100 episodes)
            final_returns = []
            for returns_list in [r for _, r in variant_results["DQN-full"]]:
                final_returns.append(np.mean(returns_list[-100:]))

            final_return_mean = np.mean(final_returns)
            final_return_std = np.std(final_returns, ddof=1)

            # Approximate steps to solve (100-episode moving average >= 475)
            steps_to_solve = "Did not solve"
            for s_steps, s_returns in variant_results["DQN-full"]:
                s_returns_array = np.array(s_returns)
                if len(s_returns_array) >= 100:
                    moving_avg = np.convolve(
                        s_returns_array, np.ones(100) / 100, mode="valid"
                    )
                    solve_idx = np.where(moving_avg >= 475.0)[0]
                    if len(solve_idx) > 0:
                        # Map episode index back to steps
                        solve_ep = solve_idx[0] + 99
                        steps_to_solve = s_steps[solve_ep]
                        break  # Found it on one seed, good enough for baseline metric

            # Save summary table
            summary_data = {
                "Algorithm": ["DQN"],
                "Steps to solve": [steps_to_solve],
                "Final return (mean)": [round(final_return_mean, 2)],
                "Std": [round(final_return_std, 2)],
                "Wall-clock (min)": [round(total_time_min, 2)],
            }
            pd.DataFrame(summary_data).to_csv(
                "results/dqn_comparing_table.csv", index=False
            )
            print("Saved DQN-full logs and comparison metrics.")

    # --- Plotting Phase ---
    print("\nGenerating Plots...")

    # Plot 1: Epsilon Decay
    plt.figure(figsize=(8, 4))
    plt.plot(epsilon_history, color="purple", linewidth=2)
    plt.title("Epsilon Decay over Training Steps")
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Epsilon Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/task2_dqn_epsilon.png")

    # Plot 2: Ablation Study
    plt.figure(figsize=(10, 6))

    # Define a common step grid for interpolation (0 to max observed steps)
    max_step_across_all = max(
        [max(steps) for var in variant_results.values() for steps, _ in var]
    )
    common_step_grid = np.linspace(0, max_step_across_all, 500)

    colors = {"DQN-full": "blue", "DQN-noReplay": "red", "DQN-noTarget": "green"}

    for variant_name, runs in variant_results.items():
        interpolated_returns = []

        for steps, returns in runs:
            # Smooth the returns with a moving average of 20 episodes before interpolating
            smoothed_returns = np.convolve(returns, np.ones(20) / 20, mode="same")
            # Interpolate onto the common step grid
            interp_ret = np.interp(common_step_grid, steps, smoothed_returns)
            interpolated_returns.append(interp_ret)

        interp_array = np.array(interpolated_returns)
        mean_ret = np.mean(interp_array, axis=0)
        std_ret = np.std(interp_array, axis=0, ddof=1)

        plt.plot(
            common_step_grid, mean_ret, label=variant_name, color=colors[variant_name]
        )
        plt.fill_between(
            common_step_grid,
            mean_ret - std_ret,
            mean_ret + std_ret,
            color=colors[variant_name],
            alpha=0.2,
        )

    plt.title("DQN Ablation Study: Return vs Total Steps")
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Return (20-Episode Moving Average)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/task2_dqn_ablation.png")

    print("Saved ablation and epsilon plots to results/ directory.")


if __name__ == "__main__":
    run_experiment()
