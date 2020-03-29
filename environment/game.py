from random import randint
from copy import copy
import random
from board import Triangle, Diamond

random.seed(1337)


class Game:
    """
    Interface for the two games, including common methods
    """

    def __init__(self, cfg, verbose):
        self.verbose = verbose
        self.size = cfg["board"]["size"]
        self.initial_player = cfg["game"]["p"]
        self.player = self.set_initial_player()

    def game_result(self):
        return 1 if self.player == 1 else -1

    def set_initial_player(self):
        player = copy(self.initial_player)
        if player not in [1, 2, 3]:
            raise Exception(
                "That is not a valid start player, please choose 1, 2 or 3. You chose {}".format(
                    self.player
                )
            )
        if player == 3:
            player = randint(1, 2)
        return player

    def change_player(self):
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def set_player(self, player):
        self.player = player

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
        pass

    def game_over(self, state):
        """
        :return: boolean
        """
        pass

    def perform_action(self, state, action):
        """
        :return: new game state
        """
        pass

    def generate_child_states(self, state):
        """
        :return: List containing tuples with all child states of given state
                 and the action taken from state to child state
                 [(child state, action to child state)]
        """
        pass
