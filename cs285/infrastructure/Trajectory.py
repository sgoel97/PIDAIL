class Trajectory:
    similarity_threshold = 0.4999988  # FIXME: currently arbitrary

    def __init__(self):
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def add_transition(self, state):
        self.transitions.append(state)

    def get_transition(self, idx):
        return self.transitions[idx]

    def compare(self, other):
        n = len(self)
        m = len(other)
        self_states = []
        other_states = []
        for i in range(n):
            for j in range(m):
                similarity = self.get_transition(i).compare(other.get_transition(j))
                if similarity > Trajectory.similarity_threshold:
                    self_states.append(i)
                    other_states.append(j)
        return self_states, other_states
