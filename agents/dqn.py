import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.networks import QNetwork


class DQNAgent:
    """
    Deep Q-Network Agent.

    LITERATURE NOTE:
    DQN bridged the gap between high-dimensional sensory inputs and reinforcement learning.
    Paper: "Human-level control through deep reinforcement learning" (Mnih et al., Nature 2015)
    Link: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        target_update_freq: int = 200,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.device = device
        self.step_counter = 0

        # 1. Initialize the Main Q-Network
        # Requirement: At least two hidden layers with ReLU (Handled in utils/networks.py)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)

        # 2. Initialize the Frozen Target Network
        # Requirement: Frozen copy, hard-updated every C steps
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Freeze target network parameters so PyTorch doesn't waste time tracking gradients for it
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 3. Loss Function
        # Requirement: MSE or Huber loss. We use Huber (SmoothL1) for gradient stability.
        self.loss_fn = nn.SmoothL1Loss()

    def get_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Requirement: Epsilon-greedy action selection.
        Note: The decay of epsilon is managed by the training loop, not the agent itself.
        """
        # Explore: Choose a random action
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        # Exploit: Choose the best action according to the Q-network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            # max(1) returns (values, indices), we want the index [1] of the max value, then extract the int
            action = q_values.max(1)[1].item()
        return action

    def update(self, replay_buffer, batch_size: int = 64):
        """
        Samples a batch from the replay buffer and performs one gradient descent step.
        """
        # Do not train if the buffer doesn't have enough samples yet
        if replay_buffer.size < batch_size:
            return

        # 1. Sample mini-batch from Replay Buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 2. Calculate Current Q-Values
        # Pass states through network, then gather the Q-values corresponding to the actions actually taken
        current_q_values = self.q_network(states).gather(1, actions)

        # 3. Calculate Target Q-Values using the Frozen Target Network
        with torch.no_grad():
            # max(1)[0] gets the max Q-value for each next state. unsqueeze(1) reshapes from [batch] to [batch, 1]
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)

            # TD Target Equation: r + gamma * max(Q(s', a')) * (1 - done)
            # If done is 1 (True), the future reward is 0.
            td_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 4. Calculate Loss and Optimize
        loss = self.loss_fn(current_q_values, td_targets)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping is an optional but highly recommended best practice in DQN
        # It prevents the gradients from exploding if a really bad batch is sampled
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # 5. Hard Update the Target Network
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
