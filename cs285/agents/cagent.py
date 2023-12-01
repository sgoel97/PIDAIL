from typing import Callable, Optional, Sequence, Tuple
import copy

import torch
from torch import nn, optim
import numpy as np

from networks.mlp import *
from infrastructure.misc_utils import *
from constants import *


def check_backup_type(backup_type):
    assert backup_type in [
        "doubleq",
        "min",
        "mean",
        "redq",
    ], f"{backup_type} is not a valid target critic backup type"


def check_gradient_type(gradient_type):
    assert gradient_type in [
        "reinforce",
        "reparametrize",
    ], f"{gradient_type} is not a valid type of actor gradient update"


def check_update_period(update_period, soft_update_period):
    assert (
        update_period is not None or soft_update_period is not None
    ), "Must specify either target_update_period or soft_target_update_rate"


class CAgent(nn.Module):
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
        target_critic_backup_type: str,
        soft_target_update_rate: int,
        actor_gradient_type: str,
        num_actor_samples: int,
        num_critic_updates: int,
        num_critic_networks: int,
        use_entropy_bonus: bool,
        backup_entropy: bool,
    ):
        super().__init__()
        check_backup_type(target_critic_backup_type)
        check_gradient_type(actor_gradient_type)
        check_update_period(target_update_period, soft_target_update_rate)

        self.total_steps = 0
        self.obs_dim = np.prod(obs_dim)  # in case it's not 1 dimensional
        self.ac_dim = ac_dim
        self.hidden_size = hidden_size

        self.discount = discount
        self.target_update_period = target_update_period
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay

        self.target_critic_backup_type = target_critic_backup_type
        self.soft_target_update_rate = soft_target_update_rate
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.backup_entropy = backup_entropy

        self.loss_fn = nn.MSELoss()

        self.actor = MLP(
            ac_dim=self.ac_dim,
            ob_dim=self.obs_dim,
            discrete=False,
            n_layers=2,
            layer_size=self.hidden_size,
            state_dependent_std=True,
        )
        self.actor_optimizer = optim.AdamW(self.actor.parameters())
        self.actor_lr_scheduler = optim.lr_scheduler.ConstantLR(
            self.actor_optimizer, factor=1.0
        )

        self.critics = nn.ModuleList(
            [
                StateActionCritic(self.obs_dim, self.ac_dim, 2, self.hidden_size)
                for _ in range(self.num_critic_networks)
            ]
        )
        self.target_critics = nn.ModuleList(
            [
                StateActionCritic(self.obs_dim, self.ac_dim, 2, self.hidden_size)
                for _ in range(self.num_critic_networks)
            ]
        )
        self.critic_optimizer = optim.AdamW(self.critics.parameters())
        self.critic_lr_scheduler = optim.lr_scheduler.ConstantLR(
            self.critic_optimizer, factor=1.0
        )
        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

        self.temperature = self.end_epsilon + (
            self.start_epsilon - self.end_epsilon
        ) * np.exp(-1.0 * self.total_steps / self.epsilon_decay)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        self.total_steps += 1
        self.temperature = self.end_epsilon + (
            self.start_epsilon - self.end_epsilon
        ) * np.exp(-1.0 * self.total_steps / self.epsilon_decay)
        with torch.no_grad():
            observation = from_numpy(observation)[None]

            action_distribution: torch.distributions.Distribution = self.actor(
                observation
            )
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.ac_dim), action.shape
            return to_numpy(action).squeeze(0)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        For example:
         - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
         - for double-Q, swap the critics' predictions (so each uses the other as the target).
         - for clip-Q, clip to the minimum of the two critics' predictions.

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        if self.target_critic_backup_type == "doubleq":
            # Dual Q-update trick
            # Swap target_Q1 and target_Q2
            assert self.num_critic_networks == 2

            next_qs = torch.stack([next_qs[1], next_qs[0]], dim=0)
        elif self.target_critic_backup_type == "min":
            # Clipped Q-update
            assert self.num_critic_networks == 2

            next_qs, _ = torch.min(next_qs, dim=0)
        elif self.target_critic_backup_type == "mean":
            # Mean Q-update
            next_qs = torch.mean(next_qs, dim=0)
        elif self.target_critic_backup_type == "redq":
            # Subsample update
            raise NotImplementedError
            num_min_qs = 2
            subsampled_next_qs = torch.gather(
                next_qs,
                0,
                torch.randint(
                    0,
                    self.num_critic_networks,
                    (
                        num_min_qs,
                        batch_size,
                    ),
                    device=ptu.device,
                ),
            )
            next_qs, _ = torch.min(subsampled_next_qs, dim=0)
        # ENDTODO

        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            next_qs = (
                next_qs[None]
                .expand((self.num_critic_networks, batch_size))
                .contiguous()
            )

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        (batch_size,) = reward.shape

        # Compute target values
        # Important: we don't need gradients for target values!
        with torch.no_grad():
            # TODO(student)
            """
            # Sample from the actor
            next_action_distribution: torch.distributions.Distribution = ...
            next_action = ...

            # Compute the next Q-values for the sampled actions
            next_qs = ...

            # Handle Q-values from multiple different target critic networks (if necessary)
            # (For double-Q, clip-Q, etc.)
            next_qs = self.q_backup_strategy(next_qs)

            # Compute the target Q-value
            target_values: torch.Tensor = ...
            """
            # Sample from the actor
            next_action_distribution: torch.distributions.Distribution = self.actor(
                next_obs
            )
            next_action = next_action_distribution.sample()
            next_qs = self.target_critic(
                next_obs,
                next_action,
            )
            # ENDTODO

            next_qs = self.q_backup_strategy(next_qs)

            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape

            # Compute the target Q-value
            target_values: torch.Tensor = reward[None] + self.discount * next_qs * (
                1 - 1.0 * done[None]
            )
            assert target_values.shape == (
                self.num_critic_networks,
                batch_size,
            ), target_values.shape

            if self.use_entropy_bonus and self.backup_entropy:
                next_action_entropy = self.entropy(next_action_distribution)

                next_action_entropy = (
                    next_action_entropy[None]
                    .expand((self.num_critic_networks, batch_size))
                    .contiguous()
                )
                assert (
                    next_action_entropy.shape == next_qs.shape
                ), next_action_entropy.shape
                next_qs -= self.temperature * next_action_entropy

        # Predict Q-values
        q_values = self.critic(obs, action)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        loss: torch.Tensor = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_net_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "q_value": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "target_value": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """
        return -action_distribution.log_prob(action_distribution.rsample())

    def actor_loss_reinforce(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        action_distribution: torch.distributions.Distribution = self.actor(obs)

        with torch.no_grad():
            action = action_distribution.sample(sample_shape=(self.num_actor_samples,))

            assert action.shape == (
                self.num_actor_samples,
                batch_size,
                self.ac_dim,
            ), action.shape

            q_values = self.critic(
                obs[None].repeat((self.num_actor_samples, 1, 1)), action
            )
            assert q_values.shape == (
                self.num_critic_networks,
                self.num_actor_samples,
                batch_size,
            ), q_values.shape

            # Our best guess of the Q-values is the mean of the ensemble
            q_values = torch.mean(q_values, axis=0)
            advantage = q_values

        log_probs: torch.Tensor = action_distribution.log_prob(action)
        torch.nan_to_num_(log_probs, nan=0.0, posinf=0.0, neginf=0.0)
        assert log_probs.shape == (
            self.num_actor_samples,
            batch_size,
        ), log_probs.shape

        # Compute the loss
        loss = torch.mean(-(advantage * log_probs))

        return loss, torch.mean(self.entropy(action_distribution))

    def actor_loss_reparametrize(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # Sample from the actor
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        action = action_distribution.rsample(sample_shape=(self.num_actor_samples,))
        assert action.shape == (
            self.num_actor_samples,
            batch_size,
            self.ac_dim,
        ), action.shape

        q_values = self.critic(obs[None].repeat((self.num_actor_samples, 1, 1)), action)
        assert q_values.shape == (
            self.num_critic_networks,
            self.num_actor_samples,
            batch_size,
        ), q_values.shape

        loss = torch.mean(-q_values)

        return loss, torch.mean(self.entropy(action_distribution))

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """

        if self.actor_gradient_type == "reparametrize":
            loss, entropy = self.actor_loss_reparametrize(obs)
        elif self.actor_gradient_type == "reinforce":
            loss, entropy = self.actor_loss_reinforce(obs)

        # Add entropy if necessary
        if self.use_entropy_bonus:
            loss -= self.temperature * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": loss.item(), "entropy": entropy.item()}

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        """
        Update the actor and critic networks.
        """

        critic_infos = []

        for _ in range(self.num_critic_updates):
            info = self.update_critic(
                observations, actions, rewards, next_observations, dones
            )
            critic_infos.append(info)

        actor_info = self.update_actor(observations)

        if (
            self.target_update_period is not None
            and self.total_steps % self.target_update_period == 0
        ):
            self.update_target_critic()
        elif self.soft_target_update_rate is not None:
            self.soft_update_target_critic(self.soft_target_update_rate)

        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }
