import numpy as np
import torch
import pickle
from sklearn.cluster import AgglomerativeClustering

from infrastructure.Transition import Transition
from infrastructure.Trajectory import Trajectory


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


def get_similar_transitions(
    trajectories, similarity_threshold=0.2, group_size_treshold=10
):
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

    grouped_transitions_filtered = np.array(
        [group for group in grouped_transitions if len(group) > group_size_treshold]
    )

    return grouped_transitions_filtered

    # num_traj = len(trajectories)
    # similar_states = []
    # for i in range(num_traj):
    #     for j in range(i + 1, num_traj):
    #         i_states, j_states = trajectories[i].compare(trajectories[j])
    #         for k in range(len(i_states)):
    #             similar_states.extend([(i, i_states[k]), (j, j_states[k])])
    # return similar_states


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
        action = agent.get_action(state.obs)
        actions.append(action)
    actions = torch.tensor(actions, dtype=torch.float32)
    return actions.var()


def get_state_collection_variance(similar_states, trajectories, agent, n_iters=100):
    """
    Gets the average variance of actions taken by our agent from a list of
    given states that are similar to eachother

    params:
        similar_states: List of states that are similar to each other
        agent: provides action for each state
        n_iters: Number of actions to take from the state when determining the variance
    returns:
        variances: variance across all states
    """
    total_variance = 0
    variances = []
    for traj_idx, state_idx in similar_states:
        state_variance = get_state_variance(
            trajectories[traj_idx].states[state_idx], agent, n_iters
        )
        total_variance += state_variance
        variances.append((traj_idx, state_idx, state_variance))
    average_variance = total_variance / len(similar_states)
    return variances
