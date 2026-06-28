#  Deep Reinforcement Learning: DQN, REINFORCE, and A2C

_Disclaimer_: opt to keep files `README.md` and `requirements.txt` in the root folder.

## Struture

The structure is a little different than what the exercise describes; we add some folders
due to the experiments on $\epsilon$ decay for the DQN agent on Task 2; noted in the tree below with a left arrow
```bash
.
├── README.md
├── agents
├── checkpoints
├── checkpoints_e_fast  <-- 
├── checkpoints_e_slow  <-- 
├── experiments
├── report.md
├── requirements.txt
├── results
├── results_e_fast <--
├── results_e_slow <--
└── utils
```

It also contains two images for showing the failure of the A2C algorithm in the CartPole-v1 environment for differnt configuration values. See report.

## Primary Environement

Suggestions (_disclaimer_: notes bases on gemini 3.5 Pro)

|Environment|State Space|Action Space|Training Wall-Time (Per Run)|Complexity & Risks |
|-----------|-----------|------------|----------------------------|-------------------|
|CartPole-v1|Continuous (4D)|Discrete (2)|~30 seconds to 2 minutes|Very Low. Extremely reliable. Ideal for verifying if code  |written from scratch is bug-free.
|LunarLander-v2|Continuous (8D)|Discrete (4)|~10 to 30 minutes|Moderate. Requires swig and box2d. Hyperparameters are sensitive;  |algorithms can easily destabilize.
|Acrobot-v1|Continuous (6D)|Discrete (3)|~2 to 5 minutes|Moderate. Non-linear dynamics can make vanilla policy gradients  |(REINFORCE) slow to converge without a baseline.

- Machine
  - CPU (i7-13800H): Excellent single-core and multi-core performance, which is the primary bottleneck for simulating simple environments like Gym/Gymnasium.
  - GPU (RTX A1000 6GB): More than sufficient for the small neural networks (MLPs) used in these environments.
- Dev environement: WSL + Python 3.12

Due to the tight timeline we choose the environement CartPole-v1, but revisit after project to see how it works in other envs.

## Setup, environment choice, how to run each task

This repository contains from-scratch implementations of three foundational Deep Reinforcement Learning algorithms: Deep Q-Networks (DQN), REINFORCE (Vanilla Policy Gradient), and Advantage Actor-Critic (A2C). 

## Environment Choice
Primary Environment: `CartPole-v1`
- State Space: Continuous (4D vector representing cart position, cart velocity, pole angle, and pole angular velocity).
- Action Space: Discrete (2 actions: push left, push right).
- Justification: CartPole-v1 was selected because it is highly reliable for validating from-scratch implementations and allows for rapid hyperparameter sweeps on a CPU/laptop GPU within a constrained timeframe. 

What CartPole does? please refer to this [video](https://www.youtube.com/watch?v=2u1REHeHMrg), on a fast tutorial.

## Project Structure
- `agents/`: Contains the core algorithmic implementations (`dqn.py`, `reinforce.py`, `a2c.py`).
- `experiments/`: Contains runnable training scripts for each task.
- `utils/`: Shared utilities including the replay buffer, neural network architectures, and plotting helpers.
- `results/`: Stores all generated training logs (`.csv`) and evaluation plots (`.png`).
- `checkpoints/`: Stores final trained model weights (`.pt`).
- `report.md`: Written analysis and empirical comparison of the algorithms.

> Note on Version Control Best Practices: > Industry best practice dictates that binary files (like `.pt` model weights and `.png` plots) and training logs (`.csv`) should be ignored by Git using a `.gitignore` file and instead tracked via tools like MLflow, Weights & Biases, or DVC. However, to strictly comply with the specific deliverables of this exercise, these files are explicitly committed to this repository.

## Setup & Installation
This project was developed on WSL running Python 3.12. To replicate the environment:

1. Create a virtual environment (venv or .venv whatever you prefer):
   ```bash
   python -m venv venv
   source venv/bin/activate
    ```

2. Install dependencies: `pip install -r requirements.txt`

## Run the experiments

- Execute all commands **from root**
- Each experiment produces results in the folder `results`, namely:
  - the plots for the deliverable in .pgn format, 
  - the logs, and
  - a file to easily compare values for task (`comparing_table.csv`)
- Additionally each experiment produces the checkpoints for each seed per agent, in the folder `checkpoints`. 

### Task 1 Tabular: 

`python -m experiments.task1_experiment`
    
This produces the following in the folder results::
- checkpoint in .npy files for each seed (see them with numpy load)

> The exercise mentions saving .pt (PyTorch) files for the final trained weights. Because Tabular Q-learning does not use neural networks, the "weights" are just the NumPy Q-table. Therefore, we save it as a .npy file. For DQN, REINFORCE, and A2C, we will use .pt files as requested.


## Task 2 DQN

`python -m experiments.task2_experiment_dqn`
  

### hyperparameter study on $\epsilon$ decay

(in the scope of task 2)

For different $\epsilon$ decaying steps we may notice different behaviours on the DQN variants by judging the ablation graphs. More on the report.

Execute with:
`python -m experiments.hyperparameter_task2.task2_experiment_dqn_e_decay_steps`

## Task 3 

### A REINFORCE

Execute with:
`python -m experiments.task3_experiment_reinforce`

#### B A2C

Execute with: `python -m experiments.task3_experiment_a2c`

## Task 4

### A. Comparison of all agents

Plot graph for all agents:
- Prerequisite: run all the experiments to create the log file for each agent.
- Run the command `python -m utils.plot_all_agents_from_files`
- Result in file `task4_comparison.png`

### B. Hyperparameter study

Execute `python -m experiments.task4_experiment_dqn_hyperparameter`

The output in the console should look like the following:
```bash
========================================
Testing Epsilon Schedule: Fast-Decay (5k)
========================================
  Training Seed 42...
  Training Seed 1337...
  Training Seed 2024...
Saved logs for Fast-Decay (5k)
DQN Fast-Decay (5k) took 2.8116132895151775

========================================
Testing Epsilon Schedule: Medium-Decay (10k)
========================================
  Training Seed 42...
  Training Seed 1337...
  Training Seed 2024...
Saved logs for Medium-Decay (10k)
DQN Medium-Decay (10k) took 3.061808156967163

========================================
Testing Epsilon Schedule: Slow-Decay (20k)
========================================
  Training Seed 42...
  Training Seed 1337...
  Training Seed 2024...
Saved logs for Slow-Decay (20k)
DQN Slow-Decay (20k) took 3.1299931248029074
```

## Brief summary

The project purpose is to undestand the main categories of deep reinforcement learning:

- value-based methods
- policy gradient methods
- actor-critic methods

------------

#### Settings

The experiments were primarily conducted on the continuous CartPole-v1 environment, with the final hyperparameter study utilizing the Acrobot-v1 environment.

The study implemented and evaluated Tabular Q-Learning, Deep Q-Networks (DQN) alongside an ablation study, REINFORCE (Monte Carlo Policy Gradient), and Advantage Actor-Critic (A2C).

#### Runnings

The algorithm of the Tabular Q-Learning was the fastest to execute at 0.28 minutes due to relying on simple dictionary lookups rather than neural networks.

ON the other hand, REINFORCE was the slowest algorithm, taking 3.07 minutes, because the agent survived for hundreds of steps per episode and required full episodic forward passes.
Notebly, A2C recorded a falsely fast time of 0.48 minutes strictly because the agent died almost immediately (every 9.4 steps) and exhausted its 1,000 allowed episodes rapidly.

Hyperparameter Runnings: In the Acrobot-v1 study, the fastest decay schedule (5k steps) was also the fastest to run in wall-clock time (2.81 minutes) and presented best results.

#### Results

None of the algorithms officially "solved" the CartPole-v1 environment, which requires a moving average of 475+.

Tabular Q-Learning was stable but sample-inefficient, DQN had the highest initial sample efficiency but collapsed later, and REINFORCE suffered from extreme variance and instability. A2C failed entirely, performing no better than random guessing (mean return of 9.4).

Most interesting feature is the DQN Ablation: The DQN-noTarget variant failed to learn completely, remaining flat near zero. The DQN-noReplay variant learned quickly but suffered from a rapid performance drop after 42,000 steps.

Hyperparameter Sweep: For the Acrobot-v1 task, the Fast-Decay (5,000 steps) schedule was the undisputed winner, recovering from the initial learning dip fastest and reaching a convergence plateau of -100 by step 15,000.

#### Insights

By exploring these set of experiments we gained insights on the following:

  - Curse of Dimensionality: The Tabular Q-Learning experiment highlights that tabular methods cannot scale to high-dimensional continuous environments because the memory and sampling requirements grow exponentially.

  - Necessity of DQN Components: The ablation study proves that a frozen target network is strictly required to prevent the divergence of the Q-function (the moving target problem), and a replay buffer is essential to decorrelate data and prevent catastrophic forgetting.

  - REINFORCE Limitations: The high variance and sample inefficiency of REINFORCE stems from its Monte Carlo nature, where gradient estimates become highly noisy by penalizing or rewarding an entire episode's worth of steps based on a single raw outcome.

  - Exploration-Exploitation Efficiency: The hyperparameter study reveals that in deterministic, low-dimensional environments with sparse rewards (like Acrobot), artificially prolonging exploration (slow epsilon decay) actively delays the onset of the optimal policy and reduces sample efficiency.

#### Future steps

DQN Tuning: The report highlights a need for further tuning of the DQN-full agent (e.g., adjusting learning rates, using a larger replay buffer, or implementing Double DQN) to prevent the sharp drops and local policy traps observed in later training steps.

A2C Investigation: Since tweaking the entropy beta and learning rate did not fix the catastrophic initialization failure of A2C, further investigation is needed to address the high gradient noise caused by 1-step TD updates.

## Extra knowledge for the author

- CartPole returns a continuous vector: `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`

## Literature for more reading investigation

- For more on the severe impact of random seeds in Deep RL, read _Deep Reinforcement Learning that Matters_,  by Henderson et al. (2018). It highlights how simply changing the random seed can dictate whether an algorithm completely fails or achieves state-of-the-art performance, underscoring why reporting multiple seeds is mandatory for empirical rigor.
- REINFORCE was introduced by Ronald J. Williams in 1992 in the seminal paper: "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Link to [paper](https://link.springer.com/article/10.1007/BF00992696).
- Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236.
- The fundamental dilemma of exploration vs. exploitation is foundational to RL. It is covered extensively in Chapter 2 of Sutton & Barto's canonical textbook: Reinforcement Learning: An Introduction. Furthermore, the $\epsilon$-greedy strategy specifically utilized in Deep Q-Networks was formalized in the original DeepMind Atari paper (Mnih et al., 2013).