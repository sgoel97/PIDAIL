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

def prune_group_action_percentile(group, percentile=1):
    """
    Only keeps states within a group that take the top percentile percentile action within the group - discrete
    """
    actions = [g["acts"] for g in group]

    numActions = len(actions)

    # manual computation because I (jg) am lazy
    counter = {}
    for action in actions:
        if action in counter:
            counter[action] += 1.0
        else:
            counter[action] = 1.0

    array = []
    for action in counter.keys():
        array.append(counter[action], action)

    array.sort(key = lambda x: -x[0])

    goodActions = {}

    cumProb = 0.0
    for prob, action in array:
        cumProb += prob * 1.0 / numActions
        goodActions[action] = 1

        if cumProb > percentile:
            break

    return [g for g in group if g["acts"] in goodActions]


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
