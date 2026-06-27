import numpy as np


class TabularQAgent:
    def __init__(
        self, state_bins: tuple, n_actions: int, lr: float = 0.1, gamma: float = 0.99
    ):
        """
        Initializes the Q-table with zeros.
        state_bins: Tuple of dimensions for the state space, e.g., (6, 6, 6, 6)
        """
        self.q_table = np.zeros(state_bins + (n_actions,))
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions

    def get_action(self, state_idx: tuple, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        # Break ties randomly to avoid getting stuck early in training
        q_values = self.q_table[state_idx]
        return np.random.choice(np.flatnonzero(q_values == q_values.max()))

    def update(
        self,
        state_idx: tuple,
        action: int,
        reward: float,
        next_state_idx: tuple,
        done: bool,
    ):
        """Updates the Q-table using the TD target."""
        best_next_action_value = (
            np.max(self.q_table[next_state_idx]) if not done else 0.0
        )
        td_target = reward + self.gamma * best_next_action_value
        td_error = td_target - self.q_table[state_idx][action]
        self.q_table[state_idx][action] += self.lr * td_error
