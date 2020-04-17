from environment.board import Diamond
from environment.game import Game


class Hex(Game):

    def __init__(self, cfg, verbose):
        super(Hex, self).__init__(cfg, verbose)
        self.board = Diamond(self.size)
        self.edges = self.get_edge_coords()
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

    def game_over(self):
        """
        :return: boolean
        """
        for path in self.paths:
            if path & self.edges[0] and path & self.edges[3]:
                if self.board.cells[path[0][0]][path[0][1]].state == (0, 1):
                    return True
            elif path & self.edges[1] and path & self.edges[2]:
                coord = next(iter(path))
                r, c = coord[0], coord[1]
                if self.board.cells[path[0][0]][path[0][1]].state == (1, 0):
                    return True
        return False

    def add_to_path(self, r, c, state):
        state = self.board.cells[r][c].state
        neighbors = self.board.get_neighbors(r, c)
        prev_path = None
        for n in neighbors:
            r_n, c_n = n[0], n[1]
            if self.board.cells[r_n][c_n].state == state:
                for path in self.paths:
                    if n in path:
                        if prev_path:
                            path.update(prev_path)
                        else:
                            path.add((r, c))
                        prev_path = path
        if not prev_path:
            newset = {(r, c)}
            self.paths.append(newset)

    def get_edge_coords(self):
        """
        """
        edge1, edge2, edge3, edge4 = set(), set(), set(), set()
        for i in range(self.size):
            edge1.add((i, 0))
            edge2.add((self.size-1, i))
            edge3.add((0, i))
            edge4.add((i, self.size-1))
        return [edge1, edge2, edge3, edge4]

    def perform_action(self, action: tuple):
        """
        :return: reward for new state
        """
        row = action[0]
        col = action[1]

        if self.player == 1:
            state = (0, 1)
        elif self.player == 2:
            state = (1, 0)

        self.board.set_cell(row, col, state)
        self.add_to_path(row, col, state)

        reward = 0

        if self.game_over():
            reward = 1000

        print(reward)
        return reward

    def generate_child_states(self, state):
        """
        :return: List containing tuples with all child states of given state
                 and the action taken from state to child state
                 [(child state, action to child state)]
        """
        pass
