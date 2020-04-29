import networkx as nx
import matplotlib.pyplot as plt
import config


class Visualizer:

    """ Creates a graphical representation of the game state """

    def __init__(self, board, size):
        self.board = board
        self.size = size

        self.node_size = config.node_size
        self.initial_color = config.initial_color
        self.p1_color = config.p1_color
        self.p2_color = config.p2_color
        self.delay = config.delay
        self.initial_edge_color = config.edge_color

        self.nodes = self.get_nodes()
        self.positions = self.get_positions()
        self.edges = self.get_edges()
        self.node_colors = self.initialize_colors()
        self.node_sizes = self.node_size
        self.graph = self.initialize_graph()
        self.edge_colors = self.get_edge_colors()

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

            for neighbor in neighbors:
                if (node, neighbor) and (neighbor, node) not in edges:
                    edges.append((node, neighbor))

        return edges

    def initialize_colors(self):
        """ Set all nodes to an initial color """

        return [self.initial_color for _ in range(len(self.nodes))]

    def initialize_edge_colors(self):
        """ Set all edges to initial color """
        return [self.initial_edge_color for _ in range(len(self.edges))]

    def initialize_graph(self):
        """ Used to create a graph object during initialization
        of Visualizer object """

        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        return G

    def fill_nodes(self, filled_nodes):
        """
        Takes in a list of node coordinates
        and colors them with the selected
        filled node color, and sets size.
        Display the new board state
        """

        p1_filled_nodes = filled_nodes[0]
        p2_filled_nodes = filled_nodes[1]

        self.node_colors = self.initialize_colors()
        self.node_sizes = [self.node_size for i in range(len(self.nodes))]

        p1_filled_indexes = [self.nodes.index(node) for node in p1_filled_nodes]
        p2_filled_indexes = [self.nodes.index(node) for node in p2_filled_nodes]

        for index in p1_filled_indexes:
            self.node_colors[index] = self.p1_color

        for index in p2_filled_indexes:
            self.node_colors[index] = self.p2_color

        self.display_board()

    def get_edge_colors(self):
        """ Set the outer edges to the color of the player that owns the edge """
        edge_colors = self.initialize_edge_colors()
        outer_edges = list(self.board.get_edge_coords())
        outer_edges_coord = []

        for i in range(len(outer_edges)):
            edges = sorted(list(outer_edges[i]))
            temp_list = []
            for j in range(len(edges) - 1):
                temp_list.append((edges[j], edges[j + 1]))

            outer_edges_coord.append(temp_list)

        for j in range(len(outer_edges_coord)):
            edge_list = outer_edges_coord[j]

            for edge in edge_list:
                edge_colors[self.edges.index(edge)] = self.player_color(j)

        return edge_colors

    def player_color(self, i):
        """ Return the color of player based on index (this is a bit hardcoded) """
        if i == 0 or i == 3:
            return self.p1_color
        else:
            return self.p2_color

    def display_board(self):
        """ Displays the board """

        plt.figure(figsize=(10, 10))

        nx.draw_networkx(
            self.graph,
            pos=self.positions,
            node_color=self.node_colors,
            node_size=self.node_sizes,
            edgecolors="black",
            edge_color=self.edge_colors,
            width=3
        )

        plt.axis("off")
        plt.show()
        plt.pause(self.delay)
