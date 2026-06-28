import numpy as np
import torch
import torch.optim as optim

from utils.networks import PolicyNetwork


class REINFORCEAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.device = device

        # 1. Policy Network (Outputs Softmax distribution implicitly via Categorical)
        self.policy_network = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Episode memory (cleared after every update)
        self.saved_log_probs = []
        self.rewards = []

    def get_action(self, state: np.ndarray) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # The network returns a PyTorch Categorical distribution
        dist = self.policy_network(state_tensor)

        # Sample an action based on the probabilities
        action = dist.sample()

        # Save the log probability of the action we just took for the loss function later
        self.saved_log_probs.append(dist.log_prob(action))

        return action.item()

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def update(self):
        """Called at the very end of an episode."""
        if len(self.rewards) == 0:
            return

        # 2. Return Computation (Discounted future rewards G_t)
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # 3. Return Normalization
        # Add a tiny epsilon (1e-8) to prevent division by zero if all returns are identical
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 4. Gradient Update: L = -E[G_t * log(pi(a_t|s_t))]
        policy_loss = []
        for log_prob, G_t in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G_t)

        # Sum the losses and perform backpropagation
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        # Clear episode memory for the next run
        self.saved_log_probs = []
        self.rewards = []
