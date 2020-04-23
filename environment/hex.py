from environment.board import Board
from environment.game import Game
from copy import deepcopy


class Hex(Game):

    def __init__(self, cfg, verbose):
        super(Hex, self).__init__(cfg, verbose)

    def generate_initial_state(self, cfg):
        """
        :return: state of the initial game, as stated in configuration file
        """
        board = Board(self.size)
        self.edges = board.get_edge_coords()
        return board

    def get_legal_actions(self, board):
        """
        :param board: current state
        :return: list of legal actions from given state
        """
        return board.get_open_cells()

    def game_over(self, board):
        """
        :return: boolean
        """
        if not len(self.get_legal_actions(board)):
            return True
        paths = self.depth_first_search(board)
        for path in paths:
            coord = next(iter(path))
            r, c = coord[0], coord[1]
            if path & self.edges[0] and path & self.edges[3]:
                if board.cells[r][c].state == (0, 1):
                    return True
            elif path & self.edges[1] and path & self.edges[2]:
                if board.cells[r][c].state == (1, 0):
                    return True
        return False

    def perform_action(self, board, action: tuple):
        """
        :return: reward for new state
        """
        row = action[0]
        col = action[1]

        if self.player == 1:
            fill = (0, 1)
        elif self.player == 2:
            fill = (1, 0)

        board.set_cell(row, col, fill)

        return board

    def generate_child_states(self, board):
        """
        :return: List containing tuples with all child states of given state
                 and the action taken from state to child state
                 [(child state, action to child state)]
        """
        children = []
        legal = self.get_legal_actions(board)
        for action in legal:
            child_state = deepcopy(board)
            child_state = self.perform_action(child_state, action)
            children.append((child_state, action))
        return children

    def depth_first_search(self, board):
        paths = []
        for row in board.cells:
            for cell in row:
                if cell.is_filled():
                    paths = self.add_to_path(
                        cell.coordinates[0], cell.coordinates[1], board, paths)
        return paths

    def add_to_path(self, r, c, board, paths):
        fill = board.cells[r][c].state
        neighbors = board.get_neighbors(r, c)
        prev_path = None
        for n in neighbors:
            r_n, c_n = n[0], n[1]
            cell = board.cells[r_n][c_n]
            if cell.state == fill:
                for path in paths:
                    if n in path:
                        if prev_path:
                            path.update(prev_path)
                        else:
                            path.add((r, c))
                        prev_path = path
        if not prev_path:
            newset = {(r, c)}
            paths.append(newset)
        return paths
