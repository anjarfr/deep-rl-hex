from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import yaml

from agent.anet import ANET
from agent.buffer import ReplayBuffer
from agent.mcts import MCTS, convert_state
from environment.hex import Hex
from environment.visualizer import Visualizer

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
        self.save_interval = cfg["nn"]["save_interval"]

        self.game = Hex(cfg, self.verbose)
        self.initial_state = self.game.generate_initial_state(cfg)
        self.sim_game = Hex(cfg, verbose=False)
        self.sim_game.generate_initial_state(cfg)
        self.state = deepcopy(self.initial_state)

        self.ANET = ANET(cfg)
        self.mcts = MCTS(cfg, self.sim_game, self.state,
                         self.simulations, self.ANET)
        self.visualizer = Visualizer(
            self.initial_state, self.game.size, cfg["display"])
        self.replay_buffer = ReplayBuffer()

    def play_game(self):
        """ One complete game """
        for i in range(self.episodes):
            self.game.change_player()

            while not self.game.game_over(self.state):
                self.game.change_player()

                """ Do simulations and choose best action """
                distribution, action = self.mcts.uct_search(self.game.player)
                print(self.game.player, action)

                # TODO: litt usikker på om det er rett å legge til self.game.player eller initial_player
                # i convert_state funk
                self.replay_buffer.add(convert_state(
                    self.state, self.game.player), distribution)

                self.state = self.game.perform_action(self.state, action)
                self.ANET.decay_epsilon()

            """ Train ANET """
            root_player = self.game.set_initial_player()
            minibatch = self.replay_buffer.create_minibatch()
            train_states = [case[0] for case in minibatch]
            train_targets = [case[1] for case in minibatch]
            self.ANET.train(train_states, train_targets)

            """ Save model parameters """
            if i % self.save_interval == 0:
                self.ANET.save(i)

            """ Reset game """
            self.mcts.reset(deepcopy(self.initial_state))
            self.state = deepcopy(self.initial_state)
            self.game.player = self.game.set_initial_player()


def main():
    player = StateManager()
    player.play_game()


if __name__ == "__main__":
    main()
