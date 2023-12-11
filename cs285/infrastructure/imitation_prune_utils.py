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
        array.append((counter[action], action))

    array.sort(key = lambda x: -x[0])

    goodActions = {}

    cumProb = 0.0
    for prob, action in array:
        cumProb += prob * 1.0 / numActions
        goodActions[action] = 1

        if cumProb > percentile / 100.0:
            break

    return [g for g in group if g["acts"] in goodActions]

def prune_group_continuous_action_percentile(group, percentile=99, agg = "or"):
    """
    prunes groups based on actions - all or any of the remaining dimensions must be within a percentile of the median. 

    """
    actions = [g["acts"] for g in group]
    action_dim = len(actions[0])
    lower_bound = np.percentile(actions, 50 - percentile // 2, axis=1)
    upper_bound = np.percentile(actions, 50 + percentile // 2, axis=1)

    assert lower_bound.shape == (action_dim, )
    assert upper_bound.shape == (action_dim, )

    if agg == "or":
        return [g for g in group if any(np.logical_and(lower_bound <= g["acts"], g["acts"] <= upper_bound))]
    elif agg == "and":
        return [g for g in group if all(np.logical_or(lower_bound <= g["acts"], g["acts"] <= upper_bound))]
    else:
        raise NotImplementedError("idk what you are doing - jg")
    
def prune_group_mean_action_distance(group, percentile=80):
    """
    prunes groups based on actions' L2 distance from the mean action of the group, using a percentile cutoff. 
    TODO implement weighted, so each coordinate is not treated the same
    """
    # print("test")
    # print(group)
    actions = [g["acts"] for g in group]
    action_dim = len(actions[0])
    mean_action = np.mean(actions, axis=0)
    assert mean_action.shape == (action_dim, )

    action_dists = [np.linalg.norm(action - mean_action) for action in actions]
    lower_bound = np.percentile(action_dists, 50 - percentile // 2)
    upper_bound = np.percentile(action_dists, 50 + percentile // 2)

    return [g for g, dist in zip(group, action_dists) if lower_bound <= dist and dist <= upper_bound]


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
