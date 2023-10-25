import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, obs_dim, hidden_dims, action_dim, loss_fn, lr=0.001):
        self.lr = lr
        self.loss_fn = loss_fn

        hidden_dims = [obs_dim] + hidden_dims + [action_dim]

        hidden_layers = []
        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(
                nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1])
            )
            hidden_layers.append(nn.ReLU())
        hidden_layers = hidden_layers[:-1]

        self.model = nn.Sequential(
            *hidden_layers,
        )

    def forward(self, obs):
        self.model.eval()
        with torch.inference_mode():
            q_values = self.model(obs)
        return q_values

    def update(self, obs, true_action):
        self.model.train()

        q_values = self.model(obs)
        loss = self.loss_fn(q_values, true_action)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
