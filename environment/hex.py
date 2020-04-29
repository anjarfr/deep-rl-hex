from environment.game import Game
from environment.static import get_neighbors


class Hex(Game):

    def __init__(self, size, player):
        super(Hex, self).__init__(size, player)

    def generate_initial_state(self, init_state=None):
        """
        :return: state of the initial game, as stated in configuration file
        """
        self.edges = self.generate_edges()
        if init_state:
            return init_state
        return [0 for _ in range(self.size**2)]

    def generate_edges(self):
        edge1, edge2, edge3, edge4 = set(), set(), set(), set()
        for i in range(self.size):
            edge1.add(i)
            edge2.add(i*self.size)
            edge3.add(i+(self.size-1)*(i+1))
            edge4.add((self.size**2-self.size)+i)
        return [edge1, edge2, edge3, edge4]

    def get_legal_actions(self, state):
        """
        :param board: current state
        :return: list of legal actions from given state
        """
        return [i for i, v in enumerate(state) if v == 0]

    def get_all_actions(self, state):
        return [i for i in range(len(state))]

    def game_over(self, state):
        """
        :return: boolean
        """
        if not len(self.get_legal_actions(state)):
            return True
        paths = self.depth_first_search(state)
        for path in paths:
            i = next(iter(path))
            if state[i] == 2:
                if path.intersection(self.edges[0]) and path.intersection(self.edges[3]):
                    return True
            if state[i] == 1:
                if path.intersection(self.edges[1]) and path.intersection(self.edges[2]):
                    return True
        return False

    def depth_first_search(self, state):
        paths = []
        for i in range(len(state)):
            if state[i] != 0:
                paths = self.add_to_path(i, state, paths)
        return paths

    def add_to_path(self, i, state, paths):
        fill = state[i]
        neighbors = get_neighbors(i, self.size)
        prev_path = None
        for n in neighbors:
            n_fill = state[n]
            if n_fill == fill:
                for path in paths:
                    if n in path:
                        if prev_path:
                            path.update(prev_path)
                        else:
                            path.add(i)
                        prev_path = path
        if not prev_path:
            newset = {i}
            paths.append(newset)
        return paths

    def perform_action(self, state, i):
        """
        :return: new state
        """
        if state[i] == 0:
            state[i] = self.player
        else:
            raise Exception(
                "Not a valid move, that cell is already occupied"
            )
        return state

    def generate_child_states(self, state):
        """
        :return: List containing tuples with all child states of given state
                 and the action taken from state to child state
                 [(child state, action to child state)]
        """
        children = []
        legal = self.get_legal_actions(state)
        for action in legal:
            child_state = self.perform_action([*state], action)
            children.append((child_state, action))
        return children
