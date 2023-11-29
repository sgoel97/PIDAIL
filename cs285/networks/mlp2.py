from typing import Optional

from torch import nn

import torch
from torch import distributions

from infrastructure.misc_utils import *
from infrastructure.distributions import make_tanh_transformed, make_multi_normal

def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
):
    """
    Builds a feedforward neural network

    arguments:
        input_placeholder: placeholder variable for the state (batch_size, input_size)
        scope: variable scope of the network

        n_layers: number of hidden layers
        size: dimension of each hidden layer
        activation: activation of each hidden layer

        input_size: size of the input layer
        output_size: size of the output layer
        output_activation: activation of the output layer

    returns:
        output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.ReLU())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp

class MLP2(nn.Module):
    """
    Base MLP policy, which can take an observation and output a distribution over actions.

    This class implements `forward()` which takes a (batched) observation and returns a distribution over actions.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        use_tanh: bool = False,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
    ):
        super().__init__()

        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std

        if discrete:
            self.logits_net = build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            )
        else:
            if self.state_dependent_std:
                assert fixed_std is None
                self.net = build_mlp(
                    input_size=ob_dim,
                    output_size=2*ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                )
            else:
                self.net = build_mlp(
                    input_size=ob_dim,
                    output_size=ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                )

                if self.fixed_std:
                    self.std = 0.1
                else:
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
                    )

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            logits = self.logits_net(obs)
            action_distribution = distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                mean, std = torch.chunk(self.net(obs), 2, dim=-1)
                std = torch.nn.functional.softplus(std) + 1e-2
            else:
                mean = self.net(obs)
                if self.fixed_std:
                    std = self.std
                else:
                    std = torch.nn.functional.softplus(self.std) + 1e-2

            if self.use_tanh:
                action_distribution = make_tanh_transformed(mean, std)
            else:
                return make_multi_normal(mean, std)

        return action_distribution

    def save(self, model_path):
        torch.save(self.net.state_dict(), model_path)

class StateActionCritic(nn.Module):
    def __init__(self, ob_dim, ac_dim, n_layers, size):
        super().__init__()
        self.net = build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        )
    
    def forward(self, obs, acs):
        return self.net(torch.cat([obs, acs], dim=-1)).squeeze(-1)

    def save(self, model_path):
        torch.save(self.net.state_dict(), model_path)