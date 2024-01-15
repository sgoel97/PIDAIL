import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, obs_dim, hidden_dims, action_dim):
        super().__init__()
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
        if type(obs) == list:
            obs = torch.stack(obs)
        if type(obs) != torch.Tensor:
            obs = torch.Tensor(obs)
        q_values = self.model(obs)
        return q_values
    
    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)