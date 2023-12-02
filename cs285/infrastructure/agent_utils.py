import constants

from stable_baselines3 import DQN, SAC


def get_default_agent(agent, discrete, default_discrete="dqn", default_continous="sac"):
    if discrete:
        agent = default_discrete if agent is None else agent
    else:
        agent = default_continous if agent is None else agent
    return agent


def get_agent(agent, env, config):
    global_config = {"tensorboard_log": "logs", "device": constants.DEVICE}

    if agent == "dqn":
        # https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
        agent = DQN("MlpPolicy", env, **config["dqn"], **global_config)
    else:
        # https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
        agent = SAC("MlpPolicy", env, **config["sac"], **global_config)
    return agent


def check_agent_env(agent, discrete, discrete_agents, continous_agents):
    if discrete:
        assert agent in discrete_agents, f"agent {agent} is not in {discrete_agents}"
    else:
        assert agent in continous_agents, f"agent {agent} is not in {continous_agents}"
