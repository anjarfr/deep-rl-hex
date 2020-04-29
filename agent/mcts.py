from agent.node import Node
from environment.static import generate_tensor_state
import random
import numpy as np
import config

random.seed(1337)


class MCTS:
    """
    Monte Carlo Tree Search
    """

    def __init__(self, game, init_state, simulations, actor):
        self.game = game
        self.root = self.create_root_node(init_state)
        self.current_node = self.root
        self.simulations = simulations
        self.c = config.c
        self.epsilon = config.mcts_epsilon
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
        the_chosen_one = self.select_move(
            self.root, c=0, stochastic=True, epsilon=self.epsilon)
        # print("Chosen", the_chosen_one.action)

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
        game_over = self.game.game_over(state)
        while not game_over:
            # If the children of the current node have not been added to the tree
            if not self.fully_expanded(self.current_node):
                self.current_node = self.node_expansion(self.current_node)
                path.append(self.current_node)
                if not self.game.game_over(state):
                    self.game.change_player()
                return path
            # If all children have been visited already, go deeper into the tree
            else:
                self.current_node = self.select_move(
                    self.current_node, self.c, stochastic=False, epsilon=0)
                path.append(self.current_node)
                state = self.current_node.state
                if not self.game.game_over(state):
                    self.game.change_player()
                else:
                    game_over = True
        return path

    def select_move(self, node: Node, c: int, stochastic: bool, epsilon):
        """ Returns the child of input node with the best Q + U value """
        legal = node.actions
        random_value = random.uniform(0, 1)

        if random_value < epsilon:
            # print("Choosing randomly!", random_value, epsilon)
            return random.choice(list(node.children.values()))

        if stochastic:
            distribution = np.array(self.get_probability_distribution())
            moves = self.game.get_all_actions(node.state)
            # print("Selecting for player ", self.game.player, distribution)
            index = np.random.choice(moves, p=distribution)
            action = moves[index]
            return node.children[action]

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
        game_over = self.game.game_over(current_state)
        while not game_over:
            current_state = self.default_policy(current_state)
            if not self.game.game_over(current_state):
                self.game.change_player()
            else:
                game_over = True
        z = self.game.game_result()
        return z

    def default_policy(self, state):
        """ Choose child state based on anet """
        children = self.game.generate_child_states(state)
        action = self.actor.choose_action(
            state=generate_tensor_state(state, self.game.player),
            legal=self.game.get_legal_actions(state),
            moves=self.game.get_all_actions(state),
            epsilon=self.actor.epsilon,
            stochastic=True
        )
        for child in children:
            if child[1] == action:
                return child[0]
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
        distribution = [0 for _ in range(self.game.size**2)]
        # Must somehow know which index corresponds to each action
        legal_actions = self.root.children.keys()
        for action in legal_actions:
            child = self.root.children.get(action)
            distribution[action] = child.visits
        return self.normalize_distribution(distribution)

    def normalize_distribution(self, distribution):
        total = sum(distribution)
        normalized_distribution = [
            float(pred) / total for pred in distribution]
        return normalized_distribution

    def reset(self, init_state):
        self.root = self.create_root_node(init_state)
        self.current_node = self.root
