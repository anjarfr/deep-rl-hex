from environment.board import Diamond
from game import Game


class Hex(Game):

    def __init__(self, cfg):
        super(Hex, self).__init__()
        self.board = Diamond(self.size)

    def generate_initial_state(self, cfg):
        """
        :return: state of the initial game, as stated in configuration file
        """
        pass

    def get_legal_actions(self, state):
        """
        :param state: current state
        :return: list of legal actions from given state
        """
        return self.board.get_open_cells()

    def game_over(self, state):
        """
        :return: boolean
        """
        pass

    def perform_action(self, state, action: tuple):
        """
        :return: new game state
        """
        row = action[0]
        col = action[1]

        if self.player == 1:
            filling = (0, 1)
        elif self.player == 2:
            filling = (1, 0)

        self.board.set_cell(row, col, filling)

        reward = 0

        if self.is_finished():
            reward = 1000

        return reward

    def generate_child_states(self, state):
        """
        :return: List containing tuples with all child states of given state
                 and the action taken from state to child state
                 [(child state, action to child state)]
        """
        pass