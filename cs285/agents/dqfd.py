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
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time


class WeightedMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weights):
        l = torch.tensor(0.0)
        for input, target, weight in zip(inputs, targets, weights):
            error = input - target
            l += error**2 * weight
        return l / weights.shape[0]


class DQfDAgent:
    def __init__(self,
                 env,
                 pretrain_steps = 10000,
                 start_epsilon = 0.9,
                 end_epsilon = 0.05,
                 epsilon_decay = 1000,
                 gamma = 0.95,
                 tau = 100,
                 sample_size = 20,
                 margin = 0.8,
                 lambda_nstep = 1.0,  # "lambda1"
                 lambda_sv = 1.0,  # "lambda2"
                 lambda_l2 = 1e-5,  # "lambda3"
                 n_step = 3):
        # Common variables
        self.env = env
        self.obs_shape = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay
        
        # Pretraining stuff
        self.pretrain_steps = pretrain_steps
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
        self.q_net = MLP(self.obs_shape, [self.hidden_dim] * 2, self.num_actions).to(DEVICE)
        self.target_net = MLP(self.obs_shape, [self.hidden_dim] * 2, self.num_actions).to(DEVICE)
        self.q_optimizer = optim.AdamW(self.q_net.parameters())
        self.q_loss_fn = WeightedMSE()
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
    
    
    def get_action(self, obs, greedy = True):
        if type(obs) != torch.Tensor:
            obs = from_numpy(obs)
        if greedy:
            prob = np.random.random()
            eps = self.end_epsilon + (self.start_epsilon - self.end_epsilon) * np.exp(
                -1.0 * self.total_steps / self.epsilon_decay
            )
            if prob > eps:
                self.q_net.eval()
                q_values = self.q_net(obs)
                action = q_values.argmax().unsqueeze(0)
                self.q_net.train()
            else:
                action = torch.Tensor([np.random.choice(range(self.num_actions))])
        else:
            self.q_net.eval()
            q_values = self.q_net(obs)
            action = q_values.argmax().unsqueeze(0)
            self.q_net.train()
        return to_numpy(action).squeeze(0).item()  # revisit: should be [a] ?
    
    
    def calc_TD(self, samples):
        obs = samples["observations"]
        actions = samples["actions"]
        if type(actions) != torch.Tensor:
            actions = torch.tensor(actions, dtype = torch.int64)
        rewards = samples["rewards"]
        next_obs = samples["next_observations"]
        dones = samples["dones"]

        self.q_net.eval()
        next_q_values = self.q_net(next_obs)
        self.q_net.train()
        best_actions = next_q_values.argmax(dim = 1)
        target_net_q_values = self.target_net(next_obs)
        target_q_values = torch.gather(target_net_q_values, 1, best_actions.unsqueeze(1)).squeeze(1)
        
        q_target = torch.tensor(rewards)
        q_target[torch.tensor(dones) != 1] += self.gamma * target_q_values[torch.tensor(dones) != 1]
        q_net_q_values = self.q_net(obs)
        q_predict = torch.gather(q_net_q_values, 1, actions.unsqueeze(1)).squeeze(1)
        return q_predict, q_target
    
    
    def calc_JE(self, samples):
        obs = samples["observations"]
        actions = samples["actions"]
        demo_infos = samples["demo_infos"]

        loss = torch.tensor(0.0)
        num_demos = 0
        for i in range(len(obs)):
            if demo_infos[i] is None:
                continue
            self.q_net.eval()
            best_actions_desc = torch.argsort(self.q_net(obs[i]), descending = True)
            self.q_net.train()
            if len(best_actions_desc) == 1:
                continue
            q_expert = self.q_net(obs[i])[actions[i]]
            best_act, second_best_act = best_actions_desc[0], best_actions_desc[1]
            max_action = second_best_act if best_act == actions[i] else best_act
            q_value = self.q_net(obs[i])[max_action]
            if q_value + self.margin < q_expert:
                continue
            else:
                loss += q_value - q_expert
                num_demos += 1
        return loss / num_demos if num_demos != 0 else loss
    
    
    def calc_Jn(self, samples, q_predict):
        demo_infos = samples["demo_infos"]

        loss = torch.tensor(0.0)
        num_demos = 0
        for i in range(len(demo_infos)):
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
            discounted_reward = reduce(lambda x, y: (x[0] + self.gamma**x[1] * y, x[1] + 1), d_rew, (0, 0))[0]
            self.q_net.eval()
            best_action = torch.argmax(self.q_net(d_next_obs))
            self.q_net.train()
            if d_done:
                target = discounted_reward
            else:
                q_value = torch.gather(self.target_net(d_next_obs), 1, best_action.unsqueeze(1)).squeeze(1)
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
        J_TD = self.q_loss_fn(q_predict, q_target, weights)  # revisit: should just be 1?
        J_E = self.calc_JE(samples)
        J_n = self.calc_Jn(samples, q_predict)
        total_loss = J_TD + self.lambda_sv * J_E + self.lambda_nstep * J_n
        total_loss.backward()
        self.q_optimizer.step()
        if self.total_steps % self.tau == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.total_steps += 1

    
    def set_log_dir(self, log_dir):
        if not os.path.exists(log_dir + "/"):
            os.makedirs(log_dir + "/")
        self.log_dir = log_dir + "/"
    
    
    def train(self, total_steps, transitions, progress_bar = False, callback = None):
        assert total_steps >= 2 * self.pretrain_steps, f"Total train steps must be at least twice pretrain steps ({self.pretrain_steps})"
        train_returns = []
        
        # Store demo transitions
        start = 0
        for i in range(len(transitions)):
            start += 1
            t = transitions[i]
            self.insert_demo_transition(t["obs"], t["acts"], t["rews"], t["next_obs"], t["dones"], i)
        self.prb.sum_tree.start = start
        
        # Pretrain on demos
        print("Pretraining on demos")
        pretrain_range = range(self.pretrain_steps)
        if progress_bar:
            pretrain_range = tqdm(pretrain_range)
        for _ in pretrain_range:
            self.update()
        
        # Train DQN
        print("Training DQN")
        train_start_time = time.time()
        obs, _ = self.env.reset()
        leftover_steps = total_steps - self.pretrain_steps
        curr_steps = 0
        total_returns = 0
        while curr_steps < leftover_steps:
            action = int(self.get_action(obs))
            next_obs, reward, done, truncated, _ = self.env.step(action)
            curr_steps += 1
            total_returns += reward
            self.insert_own_transition(obs, action, reward, next_obs, done and not truncated)
            self.update()
            if done:
                obs, _ = self.env.reset()
                train_returns.append(total_returns)
                total_returns = 0
                if progress_bar:
                    print(f"Training: {round(curr_steps / leftover_steps * 100, 2)}%, time elapsed: {format_time(time.time() - train_start_time)}", end = "\r")
                if callback:
                    callback()
            else:
                obs = next_obs
        
        # Save training rewards
        plt.plot(range(len(train_returns)), train_returns)
        plt.title("Pretraining/Training Returns")
        plt.savefig(self.log_dir + "train_returns.png")
    
    
    def evaluate(self, max_steps, n_eval_episodes = 20, new_env = None, eval = True):
        using_env = new_env if new_env else self.env
        obs, _ = using_env.reset()
        eval_returns = []
        all_num_steps = []
        print("Evaluating")
        for _ in tqdm(range(n_eval_episodes)):
            total_returns = 0
            num_steps = 0
            done = False
            while (not done) and (num_steps < max_steps):
                action = int(self.get_action(obs, greedy = False))
                next_obs, reward, done, _, _ = using_env.step(action)
                total_returns += reward
                obs = next_obs
                num_steps += 1
            obs, _ = using_env.reset()
            eval_returns.append(total_returns)
            all_num_steps.append(num_steps)
        
        if eval:
            plt.plot(range(len(eval_returns)), eval_returns)
            plt.title("Evaluation Returns")
            plt.savefig(self.log_dir + "eval_returns.png")
        return np.mean(eval_returns), np.std(eval_returns), np.mean(all_num_steps)
