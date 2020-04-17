import matplotlib.pyplot as plt
import numpy as np
import yaml
from copy import deepcopy

from environment.hex import Hex
from environment.visualizer import Visualizer
from agent.mcts import MCTS

with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


class StateManager:

    def __init__(self):
        """ initialize all the shizz """

        self.verbose = cfg['game']['verbose']
        self.episodes = cfg["agent"]["episodes"]
        self.simulations = cfg["agent"]["simulations"]
        self.anet_interval = cfg["agent"]["m"]
        self.TOPP_games = cfg["agent"]["g"]
        self.display_last_game = cfg["display"]["display_last_game"]

        self.game = Hex(cfg, self.verbose)
        self.initial_state = self.game.generate_initial_state(cfg)
        self.sim_game = Hex(cfg, verbose=False)
        self.sim_game.generate_initial_state(cfg)
        self.state = deepcopy(self.initial_state)

        self.mcts = MCTS(cfg, self.sim_game, self.state, self.simulations)
        self.visualizer = Visualizer(self.game.board, self.game.size, cfg["display"])
        # Initialize ANET with small weights and biases
        self.replay_buffer = []

    def play_game(self):

        """ One complete game """
        for i in range(self.episodes):

            while not self.game.game_over():

                """ Do simulations and choose best action """
                action = self.mcts.uct_search(self.game.player)
                self.state = self.game.perform_action(self.state, action)

            # Reset game
            self.mcts.reset(deepcopy(self.initial_state))
            self.state = deepcopy(self.initial_state)
            self.game.player = self.game.set_initial_player()


def main():
    player = StateManager()
    player.play_game()


if __name__ == "__main__":
    main()
