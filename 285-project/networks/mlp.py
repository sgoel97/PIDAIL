import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, obs_dim, hidden_dims, action_dim, loss_fn, lr=0.001):
        self.lr = lr
        self.loss_fn = loss_fn

        hidden_layers = [
            nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        ]

        self.model = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=hidden_layers[0]),
            *hidden_layers,
            nn.Linear(in_features=hidden_dims[-1], out_features=action_dim)
        )

    def forward(self, obs):
        self.model.eval()
        with torch.inference_mode():
            q_values = self.model(obs)
        return q_values

    def update(self, obs):
        self.model.train()

        q_values = self.model(obs)
        loss = self.loss_fn(q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
