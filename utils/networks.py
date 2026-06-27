import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Neural Network approximation for the Q-function: Q(s, a).

    ARCHITECTURE:
    - Input: State vector (dimension: state_dim)
    - Hidden Layer 1: Fully Connected -> 128 units -> ReLU
    - Hidden Layer 2: Fully Connected -> 128 units -> ReLU
    - Output Layer: Fully Connected -> action_dim units (one scalar Q-value per discrete action)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()

        # Requirement: At least two hidden layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Requirement: Output is one Q-value per action
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Requirement: ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # No activation function on the final layer! Q-values can be any real number.
        return self.fc3(x)
