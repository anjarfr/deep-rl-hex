from random import sample


class ReplayBuffer:
    """ Stores cases for training the ANET """

    def __init__(self):
        self.buffer = []

    def reset(self):
        self.buffer = []

    def add(self, s, D):
        """
        @param:s: state
        @param:D: target distribution
        """
        self.buffer.append((s, D))

    def create_minibatch(self):
        return sample(self.buffer, min(50, len(self.buffer)//4))
