import pickle
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from sklearn.cluster import AgglomerativeClustering

from stable_baselines3 import DQN, SAC
from stable_baselines3.common.buffers import ReplayBuffer

from imitation.data.types import TrajectoryWithRew, TransitionsWithRew

from infrastructure.imitation_prune_utils import (
    # prune_group_mode_action, # this is covered by action percentile as you can just set percentile to be epsilon
    # prune_group_vector_action, # i don't like this - jg
    prune_group_action_percentile, 
    prune_group_continuous_action_percentile, 
    prune_group_mean_action_distance, 
)
from infrastructure.custom_data_types import *


def create_imitation_trajectories(expert_file_path, custom = False):
    with open(expert_file_path, "rb") as f:
        demos = pickle.load(f)
    rollouts = []
    for demo in demos:
        if custom:
            trajectory = Trajectory()
            for i in range(len(demo) - 1):
                trajectory.transitions.append(Transition(demo["observation"][i],
                                                         demo["action"][i],
                                                         demo["observation"][i + 1],
                                                         demo["reward"][i],
                                                         i == len(demo) - 2,
                                                         i))
        else:
            trajectory = TrajectoryWithRew(
                obs=demo["observation"],
                acts=demo["action"][:-1],
                infos=None,
                terminal=True,
                rews=demo["reward"][:-1],
            )
        rollouts.append(trajectory)
    return rollouts


def group_transitions(transitions, cluster_config):
    """
    params:
        transitions: Transitions object from imitation library
    returns:
        list of (list of transitions) that are similar to each other.
    """
    # Create clustering model
    clustering_model = AgglomerativeClustering(
        n_clusters=None, **cluster_config
    )

    # Fit clustering model on transitions
    clustering_model.fit(transitions.obs)
    cluster_labels = clustering_model.labels_

    # Group transitions by cluster labels
    all_transitions = np.vstack([transitions, cluster_labels]).T
    all_transitions = all_transitions[all_transitions[:, 1].argsort()]
    grouped_transitions = np.split(
        all_transitions[:, 0],
        np.unique(all_transitions[:, 1], return_index=True)[1][1:],
    )

    return grouped_transitions


def get_transition_group_variance(transition_group, custom = False):
    """
    Gets the average variance of actions taken by expert from a given group of
    observations characterized by a transition group

    params:
        transition_group: A group of transitions with similar observations
    returns:
        variance of the actions taken
    """
    if custom:
        actions = [t.action for t in transition_group]
    else:
        actions = [t["acts"] for t in transition_group]
    return np.mean(np.var(actions, axis=-1))


def get_transition_group_entropy(transition_group, custom = False):
    """
    Gets the average variance of actions taken by expert from a given group of
    observations characterized by a transition group

    params:
        transition_group: A group of transitions with similar observations
    returns:
        variance of the actions taken
    """
    if custom:
        actions = [t.action for t in transition_group]
    else:
        actions = [t["acts"] for t in transition_group]
    counts = pd.Series(actions).value_counts().to_numpy()
    probs = torch.tensor(counts / len(transition_group))
    entropy = Categorical(probs=probs).entropy()

    return entropy.item()


def get_group_measures(transition_groups, method="variance", custom = False):
    if method == "variance":
        measure_func = get_transition_group_variance
    else:
        measure_func = get_transition_group_entropy

    measures = [measure_func(g, custom = custom) for g in transition_groups]
    return measures


def prune_transition_groups(transition_groups, discrete, prune_config):
    """
    gets rid of states within a group (keeps number of groups the same)
    """
    size_threshold, measure_cutoff, method, percentile = prune_config.values()

    measures = get_group_measures(transition_groups, method=method)
    measure_threshold = np.percentile(measures, measure_cutoff)

    measure_mask = np.array(measures) > measure_threshold

    group_sizes = np.array(list(map(len, transition_groups)))
    group_size_mask = group_sizes > size_threshold

    valid_transition_groups = []
    for i in range(len(transition_groups)):
        g = transition_groups[i]
        if measure_mask[i] and group_size_mask[i]:
            if discrete: # TODO add other pruning methods
                # g = prune_group_mode_action(g)
                g = prune_group_action_percentile(g, percentile = percentile)
            else:
                # g = prune_group_vector_action(g, percentile = percentile)
                g = prune_group_mean_action_distance(g, percentile = percentile)
        valid_transition_groups.append(g)

    return valid_transition_groups


def prune_groups_by_value(env, transitions, groups, discrete, prune_config):
    """
    Calculates Q-values of demonstrated actions
    Gets rid of samples WITHIN a group
    """
    replay_buffer = ReplayBuffer(100000, env.observation_space, env.action_space)
    for t in transitions:
        replay_buffer.add(t.obs, t.next_obs, t.action, t.reward, t.done, [])
    if discrete:
        q_estimator = DQN(policy = "MlpPolicy", env = env,
                          exploration_initial_eps = 0.0, exploration_final_eps = 0.0, learning_starts = 0)
        q_estimator.replay_buffer = replay_buffer
    else:
        q_estimator = SAC(policy = "MlpPolicy", env = env, ent_coef = 0.001)
        q_estimator.replay_buffer = replay_buffer
    q_estimator.learn(total_timesteps = np.ceil(len(transitions) / q_estimator.batch_size))
    
    masks = {}
    for group in groups:
        if discrete:
            best_q_value = float("-inf")
            best_action = None
            for transition in group:
                q_value = q_estimator.policy.forward(transition.obs)
                if q_value > best_q_value:
                    best_action = transition.action
                    best_q_value = q_value
            include_mask = np.array([t.action == best_action for t in group])
        else:
            q_values = []
            for transition in group:
                q_value = q_estimator.policy.forward(transition.obs)
                q_values.append(q_values)
            threshold = np.percentile(q_values, prune_config["cutoff"])
            include_mask = np.array(q_values) >= threshold
        masks[group] = include_mask
        
    valid_transition_groups = []
    for i in range(len(groups)):
        valid_transitions = [groups[i][j] for j in range(len(groups[i])) if masks[groups[i]][j]]
        valid_transition_groups.append(valid_transitions)       
    return valid_transition_groups 


def filter_groups_by_outcome(rollouts, groups, prune_config):
    """
    For each group of similar states, move forward `horizon` steps to get to outcome states
    For each group of outcome states, remove those that have too high variance/entropy
    Deletes entire transition groups
    """
    horizon, metric, cutoff = prune_config.values()
    outcome_groups = []
    for group in groups:
        outcomes = []
        for transition in group:
            traj_idx = transition.traj_idx
            trans_idx = transition.trans_idx
            try:
                outcomes.append(rollouts[traj_idx].transitions[trans_idx + horizon])
            except IndexError:
                outcomes.append(rollouts[traj_idx].transitions[-1])
        outcome_groups.append(outcomes)
    outcome_measures = get_group_measures(outcome_groups, method = metric, custom = True)
    measure_threshold = np.percentile(outcome_measures, cutoff)
    measure_mask = np.array(outcome_measures) < measure_threshold
    resulting_groups = [groups[i] for i in range(len(groups)) if measure_mask[i]]
    return resulting_groups


def filter_transition_groups(transition_groups, prune_config):
    """
    Deletes entire transition groups
    """
    size_threshold, measure_cutoff, method = prune_config.values()

    measures = get_group_measures(transition_groups, method=method)
    measure_threshold = np.percentile(measures, measure_cutoff)
    measure_mask = np.array(measures) < measure_threshold

    group_sizes = np.array(list(map(len, transition_groups)))
    group_size_mask = group_sizes < size_threshold

    valid_transition_groups = [
        transition_groups[i]
        for i in range(len(transition_groups))
        if (measure_mask[i] or group_size_mask[i])
    ]

    return valid_transition_groups


def collate_transitions(transition_groups):
    """
    Transform list of transition groups into one batched TransitionsWithRew object
    """
    transitions = []
    for group in transition_groups:
        transitions.extend(group)

    observations = np.array([t["obs"] for t in transitions])
    actions = np.array([t["acts"] for t in transitions])
    infos = np.array([t["infos"] for t in transitions])
    next_observations = np.array([t["next_obs"] for t in transitions])
    dones = np.array([t["dones"] for t in transitions])
    rewards = np.array([t["rews"] for t in transitions])

    transitions = TransitionsWithRew(
        observations, actions, infos, next_observations, dones, rewards
    )

    return transitions


def print_group_stats(transition_group):
    """
    prints stats about a list of transition groups
    """
    avg_group_size = np.mean(list(map(len, transition_group)))
    num_transitions = sum(map(len, transition_group))
    print(f"Number of transition groups: {len(transition_group)}")
    print(f"Number of transitions: {num_transitions}")
    print(f"Average group size: {avg_group_size}\n")
