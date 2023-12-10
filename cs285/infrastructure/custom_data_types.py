class Transition:
    def __init__(self, obs, action, next_obs, reward, done, idx):
        self.obs = obs
        self.action = action
        self.next_obs = next_obs
        self.reward = reward
        self.done = done
        self.trans_idx = idx
        self.traj_idx = None


class Trajectory:
    def __init__(self):
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def add_transition(self, transition):
        self.transitions.append(transition)

    def get_transition(self, idx):
        return self.transitions[idx]
    
    @staticmethod
    def flatten_trajectories(trajectories):
        all_transitions = []
        for i in range(len(trajectories)):
            traj = trajectories[i]
            for trans in traj.transitions:
                trans.traj_idx = i
            all_transitions.extend(transitions)
        return all_transitions
