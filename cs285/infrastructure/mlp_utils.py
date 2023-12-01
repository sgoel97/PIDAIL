import torch
import torch.nn as nn


def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    hidden_size: int,
    activation: str = "ReLU",
):
    """
    Builds a feedforward neural network

    arguments:
        input_size: size of the input layer
        output_size: size of the output layer

        n_layers: number of hidden layers
        size: dimension of each hidden layer
        activation: activation of each hidden layer

    returns:
        mlp: an FCFF neural network
    """
    return MLP(input_size, output_size, n_layers, hidden_size, activation)


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_layers: int,
        hidden_size: int,
        activation: str = "ReLU",
    ):
        super().__init__()
        layers = []
        in_size = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(getattr(nn, activation)())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def save(self, model_path):
        torch.save(self.mlp.state_dict(), model_path)
