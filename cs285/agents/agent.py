import numpy as np
import torch
import torch.nn as nn
from torch import optim

from infrastructure.mlp_utils import *
from infrastructure.misc_utils import *
from constants import *


class Agent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        ac_dim: int,
        hidden_size: int,
        discount: float,
        target_update_period: float,
        start_epsilon: float,
        end_epsilon: float,
        epsilon_decay: int,
        learning_rate: float,
    ):
        super().__init__()
        self.total_steps = 0
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.hidden_size = hidden_size
        self.discount = discount
        self.target_update_period = target_update_period
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay

        self.q_net = build_mlp(obs_dim, ac_dim, n_layers=2, hidden_size=hidden_size)
        self.target_net = build_mlp(
            obs_dim, ac_dim, n_layers=2, hidden_size=hidden_size
        )

        self.q_optimizer = optim.AdamW(self.q_net.parameters(), lr=learning_rate)
        self.q_loss_fn = nn.MSELoss()
        self.lr_schedule = optim.lr_scheduler.ConstantLR(self.q_optimizer, factor=1.0)

        self.update_target_net()

    def get_action(self, obs):
        obs = from_numpy(obs)
        epsilon = self.end_epsilon + (self.start_epsilon - self.end_epsilon) * np.exp(
            -1.0 * self.total_steps / self.epsilon_decay
        )
        self.total_steps += 1

        if np.random.random() > epsilon:
            with torch.no_grad():
                q_values = self.q_net(obs)
                action = q_values.argmax(dim=-1).unsqueeze(0)
        else:
            action = torch.tensor([np.random.choice(range(self.ac_dim))])

        return to_numpy(action).squeeze(0).item()

    def update_q_net(self, obs, actions, rewards, next_obs, dones):
        with torch.no_grad():
            target_q_value = self.target_net(next_obs).max(dim=1)[0]
            target_value = (
                rewards + self.discount * (1 - dones.type(torch.float)) * target_q_value
            )

        q_value = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.q_loss_fn(q_value, target_value)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        self.lr_schedule.step()

        return {
            "q_net_loss": loss.item(),
            "q_value": q_value.mean().item(),
            "target_value": target_value.mean().item(),
        }

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def update(self, obs, actions, rewards, next_obs, dones):
        q_net_update_info = self.update_q_net(obs, actions, rewards, next_obs, dones)

        if self.total_steps % self.target_update_period == 0:
            self.update_target_net()

        return q_net_update_info
