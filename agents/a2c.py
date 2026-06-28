import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils.networks import ActorCriticNetwork


class A2CAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_beta: float = 0.01,
        critic_coef: float = 0.5,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.beta = entropy_beta
        self.c_v = critic_coef
        self.device = device

        # 1. Network: Shared MLP backbone with Actor and Critic heads
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def get_action_and_value(self, state: np.ndarray):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist, value = self.network(state_tensor)
        action = dist.sample()

        # Return the raw tensor action for memory, the integer for the environment, and the value
        return action, action.item(), value, dist.log_prob(action), dist.entropy()

    def update(
        self, state, action_tensor, reward, next_state, done, log_prob, entropy, value
    ):
        """Performs a single-step TD update."""
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)

        # Get the Critic's estimation of the NEXT state
        with torch.no_grad():
            _, next_value = self.network(next_state_tensor)

        # 2. Advantage computation: A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        # If the episode is done, the next state has no value.
        td_target = reward_tensor + self.gamma * next_value * (1 - done_tensor)
        advantage = td_target - value

        # 3. Actor Loss (NOTE: advantage.detach() is strictly applied here per instructions)
        actor_loss = -(advantage.detach() * log_prob)

        # 4. Critic Loss: Mean squared error between predicted value and TD target
        critic_loss = F.mse_loss(value, td_target)

        # 6. Combined Loss: L_actor + c_v * L_critic - beta * Entropy
        loss = actor_loss + self.c_v * critic_loss - self.beta * entropy

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
