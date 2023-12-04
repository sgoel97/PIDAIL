import pickle
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from sklearn.cluster import AgglomerativeClustering

from infrastructure.Transition import Transition
from infrastructure.Trajectory import Trajectory

from imitation.data.types import TrajectoryWithRew


def create_imitation_trajectories(expert_file_path):
    with open(expert_file_path, "rb") as f:
        demos = pickle.load(f)
    rollouts = []
    for demo in demos:
        trajectory = TrajectoryWithRew(
            obs=demo["observation"],
            acts=demo["action"][:-1],
            infos=None,
            terminal=True,
            rews=demo["reward"][:-1],
        )
        rollouts.append(trajectory)
    return rollouts


def create_trajectories(expert_file_path):
    """
    The expert data files are formatted such that each one is a list of demos of length n = 2.
    Each demo has batch_size observations, batch_size actions, batch_size next_observations, etc.
    We define that one demo is one trajectory (which is a standard definition if we assume that
    the expert data files ARE giving us n trajectories).

    Thus the ith state along a trajectory will be made of the obs, action, next_obs, reward, and
    done located at the ith index in their respective arrays.
    """
    with open(expert_file_path, "rb") as f:
        demos = pickle.load(f)
    trajectories = []
    num_transitions = demos[0]["observation"].shape[0]
    for demo in demos:
        trajectory = Trajectory()
        for i in range(num_transitions):
            transition = Transition(
                demo["observation"][i],
                demo["action"][i],
                demo["next_observation"][i],
                demo["reward"][i],
                demo["terminal"][i],
            )
            trajectory.add_transition(transition)
        trajectories.append(trajectory)
    return trajectories


def get_similar_transitions(trajectories, similarity_threshold=0.2):
    """
    params:
        trajectories: List of trajectories with states we want to aggregate
    returns:
        list of (list of transitions) that are similar to each other.
    """
    all_transitions = np.array(
        [trajectory.transitions for trajectory in trajectories]
    ).flatten()
    all_obs = np.array([transition.obs for transition in all_transitions])

    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        metric="l2",
        linkage="average",
        distance_threshold=similarity_threshold,
    )

    # print("Fitting clustering model...")
    clustering_model.fit(all_obs)
    cluster_labels = clustering_model.labels_

    all_transitions = np.vstack([all_transitions, cluster_labels]).T
    all_transitions = all_transitions[all_transitions[:, 1].argsort()]
    grouped_transitions = np.split(
        all_transitions[:, 0],
        np.unique(all_transitions[:, 1], return_index=True)[1][1:],
    )

    return grouped_transitions


def get_transition_group_variance(transition_group):
    """
    Gets the average variance of actions taken by expert from a given group of
    observations characterized by a transition group

    params:
        transition_group: A group of transitions with similar observations
    returns:
        variance of the actions taken
    """
    actions = np.array([transition.action for transition in transition_group])
    return np.var(actions)


def get_transition_group_entropy(transition_group):
    """
    Gets the average variance of actions taken by expert from a given group of
    observations characterized by a transition group

    params:
        transition_group: A group of transitions with similar observations
    returns:
        variance of the actions taken
    """
    actions = np.array([transition.action for transition in transition_group])
    counts = pd.Series(actions).value_counts().to_numpy()
    probs = torch.tensor(counts / len(actions))
    entropy = Categorical(probs=probs).entropy()

    return entropy.item()


def get_group_measures(transition_groups, method="variance"):
    if method == "variance":
        measures = [
            get_transition_group_variance(transition_group)
            for transition_group in transition_groups
        ]

    else:
        measures = [
            get_transition_group_entropy(transition_group)
            for transition_group in transition_groups
        ]

    return measures


def filter_transition_groups(
    transition_groups, size_treshold=10, measure_cutoff=80, method="variance"
):
    assert method in ["variance", "entropy"]
    measures = get_group_measures(transition_groups, method="variance")
    measure_threshold = np.percentile(measures, measure_cutoff)
    measure_mask = np.array(measures) < measure_threshold

    group_sizes = np.array(list(map(len, transition_groups)))
    group_size_mask = group_sizes < size_treshold

    valid_transition_groups = [
        transition_groups[i]
        for i in range(len(transition_groups))
        if (measure_mask[i] or group_size_mask[i])
    ]

    return valid_transition_groups


# from infrastructure.state_utils import (
#     get_similar_states,
#     get_state_collection_variance,
#     get_action,
# )


# def get_best_action(state_collection, agent):
#     """
#     Returns best action from a collection of states using a majority-based approach from agent decisions
#     """
#     agent_actions = {}
#     for state in state_collection:
#         action = agent.get_action(state)
#         agent_actions[action] = agent_actions.get(action, 0) + 1

#     best_action = None
#     best_action_count = 0
#     for action, action_count in agent_actions.items():
#         if action_count > best_action_count:
#             best_action = action
#             best_action_count = action_count
#     return best_action


# def prune_trajectories_from_states(trajectories, similar_states, agent):
#     """
#     Prunes trajectories based on similar states to get rid of non-optimal trajectories

#     Not in-place

#     params:
#         trajectories: List of trajectories to prune
#         similar_states: List of similar states to prune trajectories from
#         agent: Agent that we use to get actions from states
#     returns:
#         pruned_trajectories: List of pruned trajectories based on similar_states input idxs
#     """
#     state_collection = []
#     for trajectory_idx, state_idx in similar_states:
#         state = trajectories[trajectory_idx][state_idx]
#         state_collection.append(state)

#     best_action = get_best_action(state_collection, agent)

#     trajectories_to_prune = []
#     for trajectory_idx, state_idx in similar_states:
#         state = trajectories[trajectory_idx][state_idx]
#         next_state = trajectories[trajectory_idx][state_idx + 1]

#         if get_action(state, next_state) != best_action:
#             trajectories_to_prune.append(trajectory_idx)

#     pruned_trajectories = trajectories.copy()
#     trajectories_to_prune.sort(reverse=True)
#     for trajectory_idx in trajectories_to_prune:
#         pruned_trajectories.pop(trajectory_idx)

#     return pruned_trajectories


# def prune_trajectories(expert_trajectories, agent, k=10):
#     """
#     Prunes expert trajectories to decrease number of high-variance states

#     params:
#         expert_trajectories: List of expert trajectories that we then prune
#         agent: Agent that we use to get actions from states
#         k: Max number of states to prune. Pruning will start by pruning the highest-variance collections of states
#     returns:
#         pruned_trajectories: List of pruned expert trajectories
#     """
#     similar_states = get_similar_states(expert_trajectories)
#     for i in range(len(similar_states)):
#         state_collection = similar_states[i]
#         state_collection_variance = get_state_collection_variance(state_collection)
#         similar_states[i] = (state_collection, state_collection_variance)

#     similar_states.sort(key=lambda x: x[1], reverse=True)

#     highest_variance_states = similar_states[:k]

#     pruned_trajectories = expert_trajectories.copy()
#     for state_collection in highest_variance_states:
#         pruned_trajectories = prune_trajectories_from_states(
#             pruned_trajectories, state_collection, agent
#         )

#     return pruned_trajectories
