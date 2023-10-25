import torch


# TODO: Maybe we use the below as a state definition?

# class State:
#     def __init__(self, obs, action, next_obs, reward, done):
#         self.obs = obs
#         self.action = action
#         self.next_obs = next_obs
#         self.reward = reward
#         self.done = done


def get_action(state, next_state):
    """
    TODO: Need to define what a state is lmao

    ideally its just (obs, action, next_obs, reward, done) but idk how trajectories are currently defined
    """
    return 0


# TODO: Refactor to work with finalized state definition


def get_similar_states(trajectories):
    """
    TODO: Aggregates similar states from a list of trajectories

    params:
        trajectories: iniiial expert trajectories to aggregate
    returns:
        similar_states: list of states that are similar to each other. This is a list of
        tuples where the first element is the index of the trajectory and the second element is the index of the observation.
    """
    return []


def get_state_variance(state, agent, n_iters=100):
    """
    Gets the variance of actions taken by our agent from a given state

    params:
        state: observation of the environment
        agent: provides action for each state
        n_iters: Number of actions to take from the state when determining the variance
    returns:
        variance: average variance of the states
    """
    actions = []
    for _ in range(n_iters):
        action = agent.get_action(state)
        actions.append(action)
    actions = torch.tensor(actions)
    return actions.var()


def get_state_collection_variance(similar_states, agent, n_iters=100):
    """
    Gets the average variance of actions taken by our agent from a list of
    given states that are similar to eachother

    params:
        similar_states: List of states that are similar to each other
        agent: provides action for each state
        n_iters: Number of actions to take from the state when determining the variance
    returns:
        variance: average variance of the states
    """
    total_variance = 0
    for state in similar_states:
        state_variance = get_state_variance(state, agent, n_iters)
        total_variance += state_variance
    average_variance = total_variance / len(similar_states)
    return average_variance
