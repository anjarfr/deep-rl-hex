import matplotlib.pyplot as plt
import numpy as np
import yaml
from copy import deepcopy

from environment.hex import Hex
from environment.visualizer import Visualizer
from agent.mcts import MCTS
from agent.buffer import ReplayBuffer
from agent.anet import ANET

with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

PATH = './models/'


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
        self.replay_buffer = ReplayBuffer()

        # Initialize ANET with small weights and biases
        self.ANET = ANET()

    def play_game(self):

        """ One complete game """
        for i in range(self.episodes):

            while not self.game.game_over():

                """ Do simulations and choose best action """
                distribution, action = self.mcts.uct_search(self.game.player)

                self.replay_buffer.add(self.state, distribution)
                self.state = self.game.perform_action(self.state, action)

            """ Train ANET """
            minibatch = self.replay_buffer.create_minibatch()
            train_states = [case[0] for case in minibatch]
            train_targets = [case[1] for case in minibatch]
            self.ANET.train(train_states, train_targets)

            """ Save model parameters """
            if i % self.anet_interval == 0:
                self.store_net()

            """ Reset game """
            self.mcts.reset(deepcopy(self.initial_state))
            self.state = deepcopy(self.initial_state)
            self.game.player = self.game.set_initial_player()


def main():
    player = StateManager()
    player.play_game()


if __name__ == "__main__":
    main()
