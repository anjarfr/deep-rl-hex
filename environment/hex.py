from environment.board import Diamond
from environment.game import Game


class Hex(Game):

    def __init__(self, cfg, verbose):
        super(Hex, self).__init__(cfg, verbose)
        self.board = Diamond(self.size)
        self.p1_edge, self.p2_edge = self.get_edge_coord()
        self.paths = []

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
        filled = self.board.get_filled_cells()
        for e in self.p1_edge:
            if self.path_is_complete(r, c):
                return True
        return False

    def search_path(self, r, c, path):
        state = self.board[r][c]
        neighbors = self.board.get_neighbors(r, c)
        for n in neighbors:
            r, c = n[0], n[1]
            if self.board[r][c].state == state:
                path.append((r,c))
                path = search_path(r, c, path)
        return path

    def path_is_complete(r, c):
        path = self.search_path(r, c, [(r, c)])
        if path[0] in self.p1_edge and path[-1] in p2_edge:
            return True
        return False

    def get_edge_coord():
        """
        """
        p1_coord = []
        p2_coord = []
        for i in range(self.size):
            p1_coord.append((i,0))
            p1_coord.append((self.size, i))
            p2_coord.append((0,i))
            p2_coord.append((i, self.size))
        return p1_coord, p2_coord

    def perform_action(self, state, action: tuple):
        """
        :return: new game state
        """
        row = action[0]
        col = action[1]

        if self.player == 1:
            state = (0, 1)
        elif self.player == 2:
            state = (1, 0)

        self.board.set_cell(row, col, state)

        reward = 0

        if self.game_over():
            reward = 1000

        return reward

    def generate_child_states(self, state):
        """
        :return: List containing tuples with all child states of given state
                 and the action taken from state to child state
                 [(child state, action to child state)]
        """
        pass