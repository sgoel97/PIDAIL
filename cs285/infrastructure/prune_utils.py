import torch
from state_utils import get_similar_states, get_state_collection_variance, get_action


def get_best_action(state_collection, agent):
    """
    Returns best action from a collection of states using a majority-based approach from agent decisions
    """
    agent_actions = {}
    for state in state_collection:
        action = agent.get_action(state)
        agent_actions[action] = agent_actions.get(action, 0) + 1

    best_action = None
    best_action_count = 0
    for action, action_count in agent_actions.items():
        if action_count > best_action_count:
            best_action = action
            best_action_count = action_count
    return best_action


def prune_trajectories_from_states(trajectories, similar_states, agent):
    """
    Prunes trajectories based on similar states to get rid of non-optimal trajectories

    Not in-place

    params:
        trajectories: List of trajectories to prune
        similar_states: List of similar states to prune trajectories from
        agent: Agent that we use to get actions from states
    returns:
        pruned_trajectories: List of pruned trajectories based on similar_states input idxs
    """
    state_collection = []
    for trajectory_idx, state_idx in similar_states:
        state = trajectories[trajectory_idx][state_idx]
        state_collection.append(state)

    best_action = get_best_action(state_collection, agent)

    trajectories_to_prune = []
    for trajectory_idx, state_idx in similar_states:
        state = trajectories[trajectory_idx][state_idx]
        next_state = trajectories[trajectory_idx][state_idx + 1]

        if get_action(state, next_state) != best_action:
            trajectories_to_prune.append(trajectory_idx)

    pruned_trajectories = trajectories.copy()
    trajectories_to_prune.sort(reverse=True)
    for trajectory_idx in trajectories_to_prune:
        pruned_trajectories.pop(trajectory_idx)

    return pruned_trajectories


def prune_trajectories(expert_trajectories, agent, k=10):
    """
    Prunes expert trajectories to decrease number of high-variance states

    params:
        expert_trajectories: List of expert trajectories that we then prune
        agent: Agent that we use to get actions from states
        k: Max number of states to prune. Pruning will start by pruning the highest-variance collections of states
    returns:
        pruned_trajectories: List of pruned expert trajectories
    """
    similar_states = get_similar_states(expert_trajectories)
    for i in range(len(similar_states)):
        state_collection = similar_states[i]
        state_collection_variance = get_state_collection_variance(state_collection)
        similar_states[i] = (state_collection, state_collection_variance)

    similar_states.sort(key=lambda x: x[1], reverse=True)

    highest_variance_states = similar_states[:k]

    pruned_trajectories = expert_trajectories.copy()
    for state_collection in highest_variance_states:
        pruned_trajectories = prune_trajectories_from_states(
            pruned_trajectories, state_collection, agent
        )

    return pruned_trajectories
