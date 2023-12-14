import time
from pathlib import Path
from datetime import timedelta

import torch

from constants import *

# from agents import *


def from_numpy(data):
    if isinstance(data, dict):
        return {k: from_numpy(v) for k, v in data.items()}
    data = torch.from_numpy(data)
    if data.dtype == torch.float64:
        data = data.float()
    return data.to(DEVICE)


def to_numpy(tensor):
    if isinstance(tensor, dict):
        return {k: to_numpy(v) for k, v in tensor.items()}
    return tensor.to("cpu").detach().numpy()


def get_env(env_name):
    env_mapper = {
        "ant": "Ant-v4",
        "cartpole": "CartPole-v1",
        "cheetah": "HalfCheetah-v4", 
        "hopper": "Hopper-v4",
        "inv_pend": "InvertedPendulum-v4",
        "lander": "LunarLander-v2",
        "pendulum": "Pendulum-v1",
        "walker": "Walker2d-v4",
    }
    return env_mapper[env_name]


def save_networks(using_demos, prune, env_name, agent):
    if using_demos:
        if prune:
            extension = "pruned"
        else:
            extension = "with_demos"
    else:
        extension = "from_scratch"

    data_path = Path(f"cs285/data/{env_name}/{int(time.time())}_{extension}/")
    data_path.mkdir(parents=True, exist_ok=True)

    try:  # lol idk how to do this
        agent.q_net.save(data_path / "q_net.pt")
        agent.target_net.save(data_path / "target_net.pt")
    except:
        for i in range(len(agent.critics)):
            agent.critics[i].save(data_path / f"critic{i}.pt")
        for i in range(len(agent.target_critics)):
            agent.target_critics[i].save(data_path / f"target_critic{i}.pt")
        agent.actor.save(data_path / "actor.pt")

    return data_path


def format_time(secs):
    td = timedelta(seconds = secs)
    return str(td)