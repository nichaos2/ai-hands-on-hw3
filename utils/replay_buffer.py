import numpy as np
import torch


class ReplayBuffer:
    """
    Experience Replay Buffer for Off-Policy RL.

    LITERATURE & STUDY LINKS:
    1. The concept of Experience Replay was originally introduced by Long-Ji Lin (1992):
       "Self-Improving Reactive Agents Based On Reinforcement Learning, Planning and Teaching"
       Link: https://apps.dtic.mil/sti/citations/ADA262601

    2. It was popularized for Deep RL by DeepMind in their canonical DQN paper (Mnih et al., 2013):
       "Playing Atari with Deep Reinforcement Learning"
       Link: https://arxiv.org/abs/1312.5602
    """

    def __init__(self, state_dim: int, max_size: int = 10000, device: str = "cpu"):
        # Requirement: Fixed-capacity circular buffer
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        # Requirement: Store (s, a, r, s', done) tuples.
        # We pre-allocate NumPy arrays instead of using a Python list of tuples.
        # This is heavily optimized for speed and memory efficiency.
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, 1), dtype=np.int64)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Adds a new transition to the buffer."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        # Circular logic: advance pointer and loop back to 0 if we hit max_size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        """
        Requirement: Sample random mini-batches.
        (We will enforce batch_size >= 32 in the DQN agent class).
        """
        # Randomly sample 'batch_size' indices from the currently populated buffer
        ind = np.random.randint(0, self.size, size=batch_size)

        # Return as PyTorch tensors pushed to the correct device (CPU/GPU)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device),
        )
