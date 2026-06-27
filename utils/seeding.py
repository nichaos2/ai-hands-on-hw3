import random

import gymnasium as gym
import numpy as np
import torch


def set_global_seed(seed: int, env: gym.Env = None):
    """
    Sets the random seed across all libraries to ensure reproducible RL runs.
    """
    # 1. Set Python core random seed
    random.seed(seed)

    # 2. Set NumPy random seed
    np.random.seed(seed)

    # 3. Set PyTorch random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic operations (may impact performance slightly but guarantees reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 4. Set Gymnasium environment and action space seeds
    if env is not None:
        # Note: In Gymnasium v0.26+, the environment seed is set during reset
        env.reset(seed=seed)
        env.action_space.seed(seed)
