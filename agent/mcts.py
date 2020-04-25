from agent.node import Node
import random
import numpy as np

#random.seed(2020)


def convert_state(state, player):
    listed_state = []
    listed_state.extend([0, 1] if player == 1 else [1, 0])
    for row in state.cells:
        for cell in row:
            listed_state.extend(list(cell.state))
    return listed_state


class MCTS:
    """
    Monte Carlo Tree Search
    """

    def __init__(self, cfg, game, init_state, simulations, actor):
        self.cfg = cfg
        self.game = game
        self.root = self.create_root_node(init_state)
        self.current_node = self.root
        self.simulations = simulations
        self.c = cfg["agent"]["c"]
        self.actor = actor

    def create_root_node(self, init_state):
        """ Initialize the root node."""
        node = Node(state=init_state, parent=None, action=None)
        return node

    def uct_search(self, player):
        """ Simulations for one move by one player in the game
        Return the child with highest score as action """

        for i in range(self.simulations):
            self.game.set_player(player)
            self.current_node = self.root
            self.simulate()

        self.game.set_player(player)
        the_chosen_one = self.select_move(self.root, c=0, stochastic=True)

        # self.root.print_tree()

        distribution = self.get_probability_distribution()
        self.root = the_chosen_one

        return distribution, the_chosen_one.action

    def simulate(self):
        """ Use tree search to find current simulation root
        Do a simulation from this root and update it and its parent's
        value based on the finite state, z, from rollout """

        path = self.tree_search()
        z = self.leaf_evaluation()
        self.backpropagate(path, z)

    def fully_expanded(self, node: Node):
        """
        Returns whether the current node has expanded all its possible children
        """
        actions = len(self.game.get_legal_actions(node.state))
        children = len(node.children)

        expanded = actions == children
        return expanded

    def node_expansion(self, node):
        """ Choose random child node to expand, and insert to tree
        Return this child
        """
        # Find out which children of the node have not been visited
        child_states = self.game.generate_child_states(node.state)
        existing_child_actions = list(node.children.keys())
        missing_child_actions = []

        for sap in child_states:
            action = sap[1]
            if action not in existing_child_actions:
                state = sap[0]
                missing_child_actions.append((state, action))

        # Choose randomly between unvisited children
        chosen = random.choice(missing_child_actions)
        # Here we expand with only this chosen child
        node.expand(chosen[0], chosen[1])

        return node.children[chosen[1]]

    def tree_search(self):
        """ Find a leaf node from the current root
        node and return the path to it """

        state = self.current_node.state
        path = [self.current_node]
        self.game.change_player()

        while not self.game.game_over(state):
            self.game.change_player()
            # If the children of the current node have not been added to the tree
            if not self.fully_expanded(self.current_node):

                self.current_node = self.node_expansion(self.current_node)
                state = self.current_node.state
                path.append(self.current_node)
                return path

            # If all children have been visited already, go deeper into the tree
            else:
                self.current_node = self.select_move(
                    self.current_node, self.c, stochastic=False)
                path.append(self.current_node)
                state = self.current_node.state

        return path

    def select_move(self, node: Node, c: int, stochastic: bool):
        """ Returns the child of input node with the best Q + U value """

        legal = node.actions

        if stochastic:
            distribution = np.array(self.get_probability_distribution())
            moves = self.root.state.get_cell_coord()
            index = np.random.choice(
                [i for i in range(len(moves))], p=distribution)
            action = moves[index]
            return self.root.children[action]

        chosen_key = random.choice(list(node.children.keys()))
        chosen = node.children[chosen_key]
        best_value = chosen.Q() + chosen.U(c)

        if self.game.player == 1:
            for action in legal:
                current_node = node.children[action]
                current_value = current_node.Q() + current_node.U(c)

                if current_value > best_value:
                    chosen = current_node
                    best_value = current_value
        else:
            for action in legal:
                current_node = node.children[action]
                current_value = current_node.Q() - current_node.U(c)

                if current_value < best_value:
                    chosen = current_node
                    best_value = current_value

        return chosen

    def leaf_evaluation(self):
        """ Perform a rollout
        Random simulation until termination
        Return end state, z """
        current_state = self.current_node.state

        while not self.game.game_over(current_state):
            new_state = self.default_policy(current_state)
            current_state = new_state[0]

            if not self.game.game_over(current_state):
                self.game.change_player()

        z = self.game.game_result()
        return z

    def default_policy(self, state):
        """ Choose a random child state """
        children = self.game.generate_child_states(state)
        action = self.actor.choose_action(
            state=state.get_board_state_as_list(self.game.player),
            legal_actions=self.game.get_legal_actions(state),
            all_actions=state.get_cell_coord(),
            epsilon=self.actor.epsilon,
            stochastic=True
        )
        for child in children:
            if child[1] == action:
                return child
        return None

    def backpropagate(self, path: list, z: int):
        """
        Update visits and average wins
        """
        for node in path:
            node.visits += 1
            node.avg_wins += z

    def get_probability_distribution(self):
        """ Return distribution child node visit counts from current root"""

        dist_length = self.game.size ** 2
        distribution = [0] * dist_length

        # Must somehow know which index corresponds to each action
        children = self.root.children
        valid_actions = self.root.children.keys()
        all_actions = self.root.state.get_cell_coord()

        for action in valid_actions:
            child = children.get(action)
            visit_count = child.visits
            action_index = all_actions.index(action)
            distribution[action_index] = visit_count

        normalized_distribution = self.normalize_distribution(distribution)
        return normalized_distribution

    def normalize_distribution(self, distribution):
        total = sum(distribution)
        normalized_distribution = [
            float(pred) / total for pred in distribution]

        return normalized_distribution

    def reset(self, init_state):
        self.root = self.create_root_node(init_state)
        self.current_node = self.root
