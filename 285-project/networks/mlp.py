import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, obs_dim, hidden_dims, num_layers, action_dim):
        hidden_layers = [
            nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1])
            for i in range(num_layers - 1)
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
    
    def save(self, model_path):
        torch.save(self.model.state_dict(), f"{model_path}.pt")