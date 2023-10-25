import torch
from torch import nn, optim
import numpy as np
from networks.mlp import MLP
from infrastructure.utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        self.hidden_dim = 64
        self.num_layers = 2
        self.discount = 0.99
        self.target_update_period = 1000
        self.start_epsilon = 0.9
        self.end_epsilon = 0.05
        self.epsilon_decay = 1000

        self.q_net = MLP(obs_shape, [self.hidden_dim] * 2, self.num_layers, num_actions).to(device)
        self.target_net = MLP(obs_shape, [self.hidden_dim] * 2, self.num_layers, num_actions).to(device)
        self.q_optimizer = optim.AdamW(self.q_net.parameters())
        self.q_loss_fn = nn.MSELoss()
        self.lr_schedule = optim.lr_scheduler.ConstantLR(self.q_optimizer, factor = 1.0)

        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.total_steps = 0

        self.update_target_net()

    
    def get_action(self, obs):
        prob = np.random.random()
        epsilon = self.end_epsilon + (self.start_epsilon - self.end_epsilon) * np.exp(-1.0 * self.total_steps / self.epsilon_decay)
        self.total_steps += 1
        if prob > epsilon:
            with torch.no_grad():
                action = self.q_net(obs).argmax(dim = 1)
        else:
            action = torch.tensor([np.random.choice(range(self.num_actions))])
        return to_numpy(action).squeeze(0).item()
    

    def update_q_net(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            target_q_value = torch.max(self.target_net(next_obs), dim = 1)[0]
            target_value = reward + self.discount * (1 - done.to(torch.float)) * target_q_value
        
        self.q_net.model.train()
        q_value = torch.gather(self.q_net(obs), 1, action.unsqueeze(1)).squeeze(1)
        loss = self.q_loss_fn(q_value, target_value)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        self.lr_schedule.step()

        return {
            "q_net_loss": loss.item(),
            "q_value": q_value.mean().item(),
            "target_value": target_value.mean().item()
        }
    

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
    

    def update(self, obs, action, reward, next_obs, done):
        q_net_update_info = self.update_q_net(obs, action, reward, next_obs, done)
        if self.total_steps % self.target_update_period == 0:
            self.update_target_net()
        return q_net_update_info