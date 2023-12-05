import sys
import torch
from torch import nn, optim
import numpy as np
from networks.mlp import MLP
from constants import *
from infrastructure.buffer import PrioritizedReplayBuffer
from infrastructure.misc_utils import *
from collections import defaultdict
import math
from functools import reduce


class DQfDAgent:
    def __init__(self,
                 obs_shape,
                 num_actions,
                 start_epsilon = 0.9,
                 end_epsilon = 0.05,
                 epsilon_decay = 1000,
                 gamma = 0.9,
                 tau = 100,
                 sample_size = 20,
                 margin = 0.8,
                 lambda_nstep = 1.0,
                 lambda_sv = 1.0,
                 lambda_l2 = 1e-5,
                 n_step = 3):
        # Common variables
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay
        
        # Pretraining stuff
        self.lambda_nstep = lambda_nstep
        self.lambda_sv = lambda_sv
        self.lambda_l2 = lambda_l2
        self.n_step = n_step
        self.margin = margin
        self.prb = PrioritizedReplayBuffer()
        self.sample_size = sample_size
        self.demo_replay = defaultdict(list)
        
        # DQN stuff
        self.tau = tau
        self.hidden_dim = 64
        self.total_steps = 0
        self.q_net = MLP(obs_shape, [self.hidden_dim] * 2, num_actions).to(DEVICE)
        self.target_net = MLP(obs_shape, [self.hidden_dim] * 2, num_actions).to(DEVICE)
        self.q_optimizer = optim.AdamW(self.q_net.parametrs())
        self.q_loss_fn = nn.MSELoss()
        self.lr_schedule = optim.lr_scheduler.ConstantLR(self.q_optimizer, factor = 1.0)
    
    def sample_from_replay(self):
        return self.prb.sample(self.sample_size)
    
    def insert_demo_transition(self, obs, action, reward, next_obs, done, demo_idx):
        if type(obs) != torch.Tensor:
            obs = from_numpy(obs)
        if type(next_obs) != torch.Tensor:
            next_obs = from_numpy(next_obs)
        episode_replay = self.demo_replay[demo_idx]
        idx = len(episode_replay)
        self.prb.insert(obs, action, reward, next_obs, done, (demo_idx, idx))
        episode_replay.append((obs, action, reward, next_obs, done, (demo_idx, idx)))
    
    def insert_own_transition(self, obs, action, reward, next_obs, done):
        if type(obs) != torch.Tensor:
            obs = from_numpy(obs)
        if type(next_obs) != torch.Tensor:
            next_obs = from_numpy(next_obs)
        self.prb.insert(obs, action, reward, next_obs, done, None)
    
    def get_action(self, obs):
        if type(obs) != torch.Tensor:
            obs = from_numpy(obs)
        prob = np.random.random()
        eps = self.end_epsilon + (self.start_epsilon - self.end_epsilon) * np.exp(
            -1.0 * self.total_steps / self.epsilon_decay
        )
        if prob > eps:
            with torch.no_grad():
                q_values = self.q_net(obs)
                action = q_values.argmax().unsqueeze(0)
        else:
            action = torch.Tensor([np.random.choice(range(self.num_actions))])
        return to_numpy(action).squeeze(0).item()
    
    def calc_TD(self, samples):
        obs = samples["observations"]
        actions = samples["actions"]
        rewards = samples["rewards"]
        next_obs = samples["next_observations"]
        dones = samples["dones"]

        next_q_values = self.q_net(next_obs)
        best_actions = next_q_values.argmax(dim = 1)
        target_q_values = torch.gather(self.target_net(next_obs), 1, best_actions)
        
        q_target = torch.Tensor(rewards)
        q_target[torch.Tensor(dones) != 1] += self.gamma * target_q_values[torch.Tensor(dones) != 1]
        q_predict = torch.gather(self.q_net(obs), 1, actions)
        return q_predict, q_target
    
    def calc_JE(self, samples):
        obs = samples["observations"]
        actions = samples["actions"]
        demo_infos = samples["demo_infos"]

        loss = torch.Tensor(0.0)
        num_demos = 0
        for i in range(len(obs)):
            if demo_infos[i] is None:
                continue
            best_actions_desc = torch.argsort(self.q_net(obs[i]), descending = True)
            if len(best_actions_desc) == 1:
                continue
            q_expert = torch.gather(self.q_net(obs[i]), 1, actions[i].unsqueeze(1)).squeeze(1)
            best_act, second_best_act = best_actions_desc[0], best_actions_desc[1]
            max_action = second_best_act if best_act == actions[i] else best_act
            q_value = torch.gather(self.q_net(obs[i]), 1, max_action)
            if q_value + self.margin < q_expert:
                continue
            else:
                loss += q_value - q_expert
                num_demos += 1
        return loss / num_demos if num_demos != 0 else loss
    
    def calc_Jn(self, samples, q_predict):
        obs = samples["observations"]
        next_obs = samples["next_observations"]
        demo_infos = samples["demo_infos"]

        loss = torch.Tensor(0.0)
        num_demos = 0
        for i in range(len(obs)):
            if demo_infos[i] is None:
                continue
            demo_idx, idx = demo_infos[i]
            n_idx = idx + self.n_step
            l_epoch = len(self.demo_replay[demo_idx])
            if n_idx > l_epoch:
                continue
            num_demos += 1
            d_obs, d_act, d_rew, d_next_obs, d_done, _ = zip(*self.demo_replay[demo_idx][idx : n_idx])
            d_obs = d_obs[-1]
            d_act = d_act[-1]
            d_next_obs = d_next_obs[-1]
            d_done = d_done[-1]
            discounted_reward = reduce(lambda x, y: (x[0] + self.gamma ** x[1] * y, x[1] + 1), d_rew, (0, 0))[0]
            best_action = torch.argmax(self.q_net(next_obs))
            if d_done:
                target = discounted_reward
            else:
                q_value = torch.gather(self.target_net(next_obs), 1, best_action.unsqueeze(1)).squeeze(1)
                target = discounted_reward + self.gamma ** self.n_step * q_value
            pred = q_predict[i]
            loss += (target - pred) ** 2
        return loss / num_demos
    
    def update(self):
        self.q_optimizer.zero_grad()
        samples, indices, weights = self.sample_from_replay()
        q_predict, q_target = self.calc_TD(samples)
        for i in range(self.sample_size):
            error = math.fabs(float(q_predict[i] - q_target[i]))
            self.prb.update(indices[i], error)
        J_TD = self.q_loss_fn(q_predict, q_target, weights)
        J_E = self.calc_JE(samples)
        Jn = self.calc_Jn(samples, q_predict)
        total_loss = J_TD + self.lambda_sv * J_E + self.lambda_nstep * Jn
        total_loss.backward()
        self.q_optimizer.step()
        if self.total_steps % self.tau == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            self.total_steps += 1
