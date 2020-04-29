from environment.board import Board
from math import sqrt, floor


def get_cell_coord(self):
    return [i for i in range(self.size**2)]


def get_neighbors(i, size):
    """ Finds the neighbors of the cell in (r, c)
    and returns a list with their coordinates """
    r = i // size
    c = i % size

    legal = [
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
    ]

    indexes = [
        i-size,
        i-(size-1),
        i-1,
        i+1,
        i+(size-1),
        i+size
    ]

    neighbors = []
    for j, cell in enumerate(legal):
        row, col = cell[0], cell[1]
        if is_legal_cell(row, col, size):
            neighbors.append(indexes[j])
    return neighbors


def is_legal_cell(row, col, size):
    """ Checks if the cell in (row, col) is a valid one
    e.g. it is not outside of the board """
    return not (row < 0 or row > size - 1 or
                col < 0 or col > size - 1)


def generate_tensor_state(state, player):
    """Returns the state as a list of 1s and 0s"""
    buffer_state = []
    for c in state:
        if c == 1:
            buffer_state.extend([0, 1])
        elif c == 2:
            buffer_state.extend([1, 0])
        else:
            buffer_state.extend([0, 0])

    if player == 1:
        return [0, 1] + buffer_state
    else:
        return [1, 0] + buffer_state


def generate_board_state(state, size):
    """
    Generate board state from OHT server state
    :param state: (1 or 2, 0, 0, 0, 0 ...)
    """
    board = Board(size)
    r = 0
    c = 0
    for i, v in enumerate(state):
        if v == 0:
            fill = (0, 0)
        elif v == 1:
            fill = (0, 1)
        else:
            fill = (1, 0)
        board.set_cell(r, c, fill)
        c += 1
        if c > size-1:
            r += 1
            c = 0
    return board


def convert_row_col(i, size):
    return (i // size, i % size)
