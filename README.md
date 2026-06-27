#  Deep Reinforcement Learning: DQN, REINFORCE, and A2C

_Disclaimer_: opt to keep files `README.md` and `requirements.txt` in the hw3 folder.

The project purpose is to undestand the main categories of deep reinforcement learning:

- value-based methods
- policy gradient methods
- actor-critic methods


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

3. Run the experiments
