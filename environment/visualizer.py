import networkx as nx
import matplotlib.pyplot as plt


class Visualizer:
    """ Creates a graphical representation of the game state """

    def __init__(self, board, size, display_options):
        self.board = board
        self.size = size

        self.node_size = display_options["node_size"]
        self.initial_color = display_options["initial_color"]
        self.p1_color = display_options["p1_color"]
        self.p2_color = display_options["p2_color"]
        self.delay = display_options["delay"]

        self.nodes = self.get_nodes()
        self.positions = self.get_positions()
        self.edges = self.get_edges()
        self.node_colors = self.initialize_colors()
        self.node_sizes = self.node_size
        self.graph = self.initialize_graph()

    def get_nodes(self):
        """ Get the coordinates of all the cells on the board """

        return self.board.get_cell_coord()

    def get_positions(self):
        """
        Create positions for every cell
        in plot based on the cells' coordinates
        and return a dictionary with the values
        (r, c): [x, y]
        """

        positions = {}
        counter = 0

        for i in range(self.size, 0, -1):
            for j in range(self.size):
                positions[self.nodes[counter]] = [i + j, i - j]
                counter += 1

        return positions

    def get_edges(self):
        """
        Find all the edges between the cells
        and its neighbors and return a list of
        tuples on the form (node, neighbor)
        indicating an edge from node to neighbor
        """

        edges = []

        for node in self.nodes:
            neighbors = self.board.get_neighbors(node[0], node[1])
            edges += [(node, neighbor) for neighbor in neighbors]

        return edges

    def initialize_colors(self):
        """ Set all nodes to an initial color """

        return [self.initial_color for i in range(len(self.nodes))]

    def initialize_graph(self):
        """ Used to create a graph object during initialization
        of Visualizer object """

        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        return G

    def fill_nodes(self, filled_nodes, jump_from=None, jump_to=None):
        """
        Takes in a list of node coordinates
        and colors them with the selected
        filled node color, and sets size.
        Display the new board state
        """
        self.node_colors = self.initialize_colors()
        filled_indexes = [self.nodes.index(node) for node in filled_nodes]
        self.node_sizes = [self.node_size for i in range(len(self.nodes))]

        for index in filled_indexes:
            self.node_colors[index] = self.filled_color

        if jump_from != None and jump_to != None:
            start = self.nodes.index(jump_from)
            end = self.nodes.index(jump_to)
            self.node_colors[start] = self.start_color
            self.node_colors[end] = self.end_color

            self.node_sizes[start] = self.start_size
            self.node_sizes[end] = self.end_size

        self.display_board()

    def display_board(self):
        """ Displays the board """

        plt.figure(figsize=(10, 10))

        nx.draw_networkx(
            self.graph,
            pos=self.positions,
            node_color=self.node_colors,
            node_size=self.node_sizes,
            edgecolors="black",
        )

        plt.axis("off")
        plt.show()
        plt.pause(self.delay)

