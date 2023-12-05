import pickle

import numpy as np
import gymnasium as gym

import os

from tqdm import tqdm

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

ENV_NAME = "LunarLander-v2"

# Retrieve the model from the hub
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository including the extension .zip
checkpoint = load_from_hub(
    repo_id=f"sb3/ppo-{ENV_NAME}",
    filename=f"ppo-{ENV_NAME}.zip",
)
expert = PPO.load(checkpoint)

env = gym.make(ENV_NAME)

demos = []

eval_env = gym.make(ENV_NAME)
mean_reward, std_reward = evaluate_policy(
    expert, eval_env, render=False, n_eval_episodes=10, deterministic=True, warn=False
)
print(f"mean_reward={mean_reward} +/- {std_reward}")

for traj in range(3): # changeable
    observations = []
    actions = []
    rewards = []

    obs, _ = env.reset()
    # observations.append(obs)
    for i in tqdm(range(1000)): # changeable
        action, _states = expert.predict(obs, deterministic=True)
        next_obs, reward, done, info, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        if done:
            break
        else:
            obs = next_obs
        # TODO store these into pickle/parquet files

    print(len(observations), len(actions), len(rewards))
    demo = {"observation": np.array(observations), "action": np.array(actions), "reward":np.array(rewards)}
    demos.append(demo)

with open(f"{os.getcwd()}/cs285/experts/expert_data_{ENV_NAME}.pkl", "wb") as f:
    pickle.dump(demos, f)