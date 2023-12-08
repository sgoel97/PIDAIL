import numpy as np
from scipy import stats


def prune_group_mode_action(group):
    """
    Only keeps states within a group that take the mode action within a group
    """
    actions = [g["acts"] for g in group]
    majority_action = stats.mode(actions, keepdims=False).mode
    pruned_group = [g for g in group if g["acts"] == majority_action]
    return pruned_group


def prune_group_vector_action(group, percentile=50):
    """
    Creates a mean vector which averages each feature of the action vector,
    then keeps action vectors that are within a percentile of the mean action vector
    """
    actions = [g["acts"] for g in group]
    mean_action = np.mean(actions, axis=0, keepdims=True)

    action_dists = np.array(actions) - mean_action
    action_dists = np.mean(action_dists, axis=-1)

    lower_bound = np.percentile(action_dists, 50 - percentile // 2)
    upper_bound = np.percentile(action_dists, 50 + percentile // 2)

    pruned_group = []
    for i in range(len(group)):
        if lower_bound <= action_dists[i] <= upper_bound:
            pruned_group.append(group[i])

    return pruned_group
