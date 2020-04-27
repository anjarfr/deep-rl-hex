from random import sample
import collections


class ReplayBuffer:
    """ Stores cases for training the ANET """

    def __init__(self, max_size):
        self.buffer = collections.deque()
        self.max_size = max_size

    def reset(self):
        self.buffer = []

    def add(self, s, D):
        """
        @param:s: state
        @param:D: target distribution

        """
        # To keep buffer at fixed size
        self.buffer.appendleft((s, D))
        if len(self.buffer) >= self.max_size + 1:
            self.buffer.pop()

        #print("Added", (s, D), "to buffer of length", len(self.buffer))

    def show_variance(self):
        # Make a list of only states
        state_list = []
        for item in self.buffer:
            state_list.append(item[0])

        tuple_states = map(tuple, state_list)

        # Count occurrences of different states
        counter = collections.Counter(tuple_states)
        print(counter.values())
        print(len(counter))

    def create_minibatch(self, batch_size):
        return sample(self.buffer, min(len(self.buffer), batch_size))
