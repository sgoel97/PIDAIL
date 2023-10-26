class Trajectory:
    similarity_threshold = 0.4999988  # TODO: currently arbitrary

    def __init__(self):
        self.states = []

    def __len__(self):
        return len(self.states)

    def add_state(self, state):
        self.states.append(state)

    def get_state(self, idx):
        return self.states[idx]

    def compare(self, other):
        n = len(self)
        m = len(other)
        self_states = []
        other_states = []
        for i in range(n):
            for j in range(m):
                similarity = self.get_state(i).compare(other.get_state(j))
                if similarity > Trajectory.similarity_threshold:
                    self_states.append(i)
                    other_states.append(j)
        return self_states, other_states
