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

np.float_ = np.float64


def run_hyperparameter_sweep():
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Switch to the new environment
    env_name = "Acrobot-v1"
    seeds = [42, 1337, 2024]
    max_episodes = 400  # Acrobot usually takes a bit longer to solve
    batch_size = 64

    # Epsilon sweep configurations
    epsilon_start = 1.0
    epsilon_min = 0.05

    variants = {
        "Fast-Decay (5k)": {
            "save_name": "fast_dqn",
            "decay_steps": 5000,
            "color": "red",
        },
        "Medium-Decay (10k)": {
            "save_name": "moderate_dqn",
            "decay_steps": 10000,
            "color": "blue",
        },
        "Slow-Decay (20k)": {
            "save_name": "slow_dqn",
            "decay_steps": 20000,
            "color": "green",
        },
    }

    variant_results = {}

    for variant_name, config in variants.items():
        save_name = config["save_name"]
        print(f"\n{'=' * 40}\nTesting Epsilon Schedule: {variant_name}\n{'=' * 40}")
        variant_results[variant_name] = []
        log_data = []
        experiment_start_time = time.time()

        for seed in seeds:
            print(f"  Training Seed {seed}...")
            env = gym.make(env_name)
            set_global_seed(seed, env)

            # Note: Acrobot has 6 state dims and 3 action dims.
            # Our network handles this dynamically!
            agent = DQNAgent(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                lr=1e-3,
                gamma=0.99,
                target_update_freq=200,
                device="cpu",
            )

            replay_buffer = ReplayBuffer(
                state_dim=env.observation_space.shape[0], max_size=10000, device="cpu"
            )

            # --- Warm-up Phase ---
            # NOTE: Do not begin gradient updates until the buffer contains
            #  at least 1,000 transitions collected with random actions
            state, _ = env.reset(seed=seed)
            while replay_buffer.size < 1000:
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
                    # CALCULATE EPSILON BASED ON CURRENT VARIANT'S DECAY STEPS
                    decay_ratio = min(1.0, cumulative_steps / config["decay_steps"])
                    epsilon = epsilon_start - decay_ratio * (
                        epsilon_start - epsilon_min
                    )
                    epsilon = max(epsilon_min, epsilon)

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

                log_data.append(
                    {
                        "Variant": variant_name,
                        "Seed": seed,
                        "Episode": episode,
                        "Cumulative_Steps": cumulative_steps,
                        "Return": total_reward,
                    }
                )

            variant_results[variant_name].append((seed_steps, seed_returns))

            # Save checkpoint
            # safe_name = (
            #     variant_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            # )
            torch.save(
                agent.q_network.state_dict(),
                f"checkpoints/dqn_acrobot_{save_name}_seed_{seed}.pt",
            )
            env.close()

        # Save Logs for this variant
        df_logs = pd.DataFrame(log_data)
        df_logs.to_csv(f"results/dqn_acrobot_{save_name}_logs.csv", index=False)
        print(f"Saved logs for {variant_name}")

        experiment_end_time = time.time()
        total_time_min = (experiment_end_time - experiment_start_time) / 60.0
        print(f"DQN {variant_name} took {total_time_min}")
    # end of variant
    # --- Plotting Phase ---
    print("\nGenerating Hyperparameter Study Plot...")
    plt.figure(figsize=(10, 6))

    # Common step grid for all variants
    max_step_across_all = max(
        [max(steps) for var in variant_results.values() for steps, _ in var]
    )
    common_step_grid = np.linspace(0, max_step_across_all, 500)

    for variant_name, runs in variant_results.items():
        interpolated_returns = []

        for steps, returns in runs:
            # Acrobot returns are negative (e.g., -500 to -80).
            # We smooth them with a 20-episode moving average
            smoothed_returns = np.convolve(returns, np.ones(20) / 20, mode="same")
            interp_ret = np.interp(common_step_grid, steps, smoothed_returns)
            interpolated_returns.append(interp_ret)

        interp_array = np.array(interpolated_returns)
        mean_ret = np.mean(interp_array, axis=0)
        std_ret = np.std(interp_array, axis=0, ddof=1)

        color = variants[variant_name]["color"]
        plt.plot(
            common_step_grid, mean_ret, label=variant_name, color=color, linewidth=2
        )
        plt.fill_between(
            common_step_grid,
            mean_ret - std_ret,
            mean_ret + std_ret,
            color=color,
            alpha=0.2,
        )

    plt.title("DQN Epsilon Decay Sweep on Acrobot-v1")
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Return (20-Episode Moving Average)")

    # Acrobot rewards are -1 per step, so we put the legend in the bottom right to avoid covering the lines
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/task4_hyperparam_study.png")

    print("Saved hyperparameter sweep plot to results/task4_hyperparam_study.png")


if __name__ == "__main__":
    run_hyperparameter_sweep()
