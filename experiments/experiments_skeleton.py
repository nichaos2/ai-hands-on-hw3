# import numpy as np
# # ... imports for your agents and utilities ...

# def run_experiment():
#     seeds = [42, 1337, 2024]
#     all_returns = []

#     for seed in seeds:
#         print(f"--- Running training for seed {seed} ---")

#         # 1. Initialize environment
#         env = gym.make("CartPole-v1")

#         # 2. Enforce reproducibility
#         set_global_seed(seed, env)

#         # 3. Initialize Agent (DQN, REINFORCE, or A2C)
#         # agent = DQNAgent(...)

#         # 4. Train Agent
#         # seed_returns = train(agent, env)

#         # 5. Store returns for this seed
#         # all_returns.append(seed_returns)

#          env.close()

#     # all_returns is a list of lists (or a 2D NumPy array of shape [num_seeds, num_episodes])
#     # Next, we calculate mean and standard deviation across axis 0 (the seeds)
