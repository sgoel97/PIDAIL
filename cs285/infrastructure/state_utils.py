import pickle
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
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
