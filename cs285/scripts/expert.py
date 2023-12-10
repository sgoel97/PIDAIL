import pickle

import numpy as np
import gymnasium as gym

import os

from tqdm import tqdm

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.evaluation import evaluate_policy

ENV_NAME = "Ant-v4"

# Retrieve the model from the hub
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository including the extension .zip 
experts = []

names = ["ppo", "dqn", "sac"]
sb3s = [PPO, DQN, SAC]

for name, sb3 in zip(names, sb3s):
    try:
        checkpoint = load_from_hub(
            repo_id=f"sb3/{name}-{ENV_NAME}",
            filename=f"{name}-{ENV_NAME}.zip",
        )
        experts.append(sb3.load(checkpoint))
        print(f"successfully retrieved {name}")
    except:
        print(f"failed to retrieve {name}")

if len(experts) == 0:
    checkpoint = load_from_hub(
        repo_id = f"cleanrl/{ENV_NAME}-ppo_continuous_action-seed1",
        filename=f"ppo_continuous_action.cleanrl_model"
    )
    experts.append(PPO.load(checkpoint))

env = gym.make(ENV_NAME)

demos = []

for expert in experts:
    eval_env = gym.make(ENV_NAME)
    mean_reward, std_reward = evaluate_policy(
        expert, eval_env, render=False, n_eval_episodes=10, deterministic=True, warn=False
    )
    print(f"mean_reward={mean_reward} +/- {std_reward}")

num_transitions = 0
while num_transitions < 10000: # changeable
    observations = []
    actions = []
    rewards = []

    for exp in experts:
        obs, _ = env.reset()
        for i in tqdm(range(1000)): # changeable
            action, _states = exp.predict(obs, deterministic=True)
            next_obs, reward, done, info, _ = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            num_transitions += 1
            if done:
                break
            else:
                obs = next_obs

    # print(len(observations), len(actions), len(rewards))
    demo = {"observation": np.array(observations), "action": np.array(actions), "reward":np.array(rewards)}
    demos.append(demo)

with open(f"{os.getcwd()}/cs285/experts/expert_data_v2_{ENV_NAME}.pkl", "wb") as f:
    pickle.dump(demos, f)
    print(f"saved expert data to {os.getcwd()}/cs285/experts/expert_data_v2_{ENV_NAME}.pkl")