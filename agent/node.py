from math import log, sqrt, inf

class Node:

    """ Node for generating tree structure """

    def __init__(self, state, parent, action):

        """ Description
        :state:         The game state in this node. Can be any type
        :parent:        Parent Node of type Node
        :action:        The action leading from the parent to this node
        :children:      Dict with {'action': child_node} generated from this node
        :actions:       List of actions (any type) that can be taken from this node
        :visits:        Integer N(s). Number of times this node has been visited
        :t:             Integer, average total wins of games including this node
        """

        self.state = state
        self.parent = parent
        self.action = action

        self.children = {}
        self.actions = []
        self.visits = 0
        self.avg_wins = 0

    def Q(self):
        return self.avg_wins / self.visits

    def U(self, c: int):
        """ Exploration bonus """
        if self.visits == 0:
            return inf
        return c * sqrt(log(self.parent.visits)/self.visits)

    def expand(self, child_state, action_to_child):
        """ Insert a new child node into the tree. Add action to parent """
        child = Node(state=child_state, parent=self, action=action_to_child)
        self.actions.append(action_to_child)
        self.children[action_to_child] = child

    def print_tree(self):
        children = list(self.children.values())
        print("Root:", "state", self.state, "Avg wins", self.avg_wins, "visits", self.visits)

        while len(children):
            temp_children = []
            for _ in children:
                child = children.pop(0)
                print("Child: ", "state ", child.state, "parent", child.parent.state, "Avg wins ", child.avg_wins, "visits ", child.visits)
                for temp_child in list(child.children.values()):
                    temp_children.append(temp_child)

            for child in temp_children:
                children.append(child)

