import numpy as np


class Board:

    def __init__(self, size):
        self.size = size
        self.cells = self.create_board(self.size)

    def create_board(self, size):
        """ Creates a diamond board of the specified size """

        cells = np.empty((size, size), dtype="object")
        for row in range(size):
            for col in range(size):
                cells[row][col] = Cell((0, 0), (row, col))
        return cells

    def set_cell(self, r, c, state):
        """ Sets the state of cell in (r, c) to state """
        self.cells[r][c].set_state(state)

    def get_cell_coord(self):
        """ Returns a list with coordinates to
        all the cells in the board, regardless
        of their state """

        return list(zip(*np.where(self.cells != None)))

    def get_filled_cells(self):
        """ Returns a list with coordinates to
        all non-empty cells in the board
        Return two lists, one for player 1 and one for player 2
        """
        p1_filled = []
        p2_filled = []

        for row in self.cells:
            for cell in row:
                if cell.is_filled():
                    if cell.is_player1():
                        p1_filled.append(cell.coordinates)
                    else:
                        p2_filled.append(cell.coordinates)

        return p1_filled, p2_filled

    def get_open_cells(self):
        """ Returns a list with coordinates to
        all empty cells in the board """

        open_cells = []
        for row in self.cells:
            for cell in row:
                if cell != None:
                    if not cell.is_filled():
                        open_cells.append(cell.coordinates)
        return open_cells

    def get_edge_coords(self):
        edge1, edge2, edge3, edge4 = set(), set(), set(), set()

        for i in range(self.size):
            edge1.add((i, 0))
            edge2.add((self.size - 1, i))
            edge3.add((0, i))
            edge4.add((i, self.size - 1))

        return [edge1, edge2, edge3, edge4]

    def get_board_state_as_list(self, player):
        """Returns the board as a list of 1s and 0s"""
        state = []

        for row in self.cells:
            for cell in row:
                state.extend(cell.state)

        if player == 1:
            return [0, 1] + state
        else:
            return [1, 0] + state

    def get_neighbors(self, r, c):
        """ Finds the neighbors of the cell in (r, c)
        and returns a list with their coordinates """

        tmp = [
            (r - 1, c),
            (r - 1, c + 1),
            (r, c - 1),
            (r, c + 1),
            (r + 1, c - 1),
            (r + 1, c),
        ]
        neighbors = []
        for cell in tmp:
            row = cell[0]
            col = cell[1]
            if self.is_legal_cell(row, col):
                neighbors.append(cell)
        return neighbors

    def is_legal_cell(self, row, col):
        """ Checks if the cell in (row, col) is a valid one
        e.g. it is not outside of the board """

        return not (row < 0 or row > self.size - 1 or col < 0 or col > self.size - 1)


class Cell:
    """
    Represents a cell in the board. The state can have the values
    Peg: (0,0) - empty, (0,1) - filled
    Hex: (0,0) - emtpy, (0,1) - player 1, (1,0) - player 2
    """

    def __init__(self, state, coordinates):
        self.state = state
        self.visited = False
        self.coordinates = coordinates

    def set_visited(self, visited):
        """ Mark a cell as visited during search """

        self.visited = visited

    def set_state(self, state):
        """ Alter the cell's state """

        self.state = state

    def is_filled(self):
        """ Checks whether the cell is empty """

        return self.state != (0, 0)

    def is_player1(self):
        return self.state == (0, 1)

