from copy import deepcopy
import yaml
import matplotlib.pyplot as plt

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
        self.sim_game_state = self.sim_game.generate_initial_state(cfg)
        self.state = deepcopy(self.initial_state)

        self.ANET = ANET(cfg)
        self.mcts = MCTS(cfg, self.sim_game, self.sim_game_state,
                         self.simulations, self.ANET)
        self.visualizer = Visualizer(
            self.initial_state, self.game.size, cfg["display"])
        self.replay_buffer = ReplayBuffer()

    def print_loss_and_accuracy(self, loss, accuracy):
        plt.plot(loss)
        plt.ylabel('Loss')
        plt.plot(accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('Iteration')
        plt.show()

    def play_game(self):
        """ One complete game """
        for i in range(self.episodes):
            self.game.set_initial_player()

            while not self.game.game_over(self.state):
                """ Do simulations and choose best action """
                distribution, action = self.mcts.uct_search(self.game.player)
                current_state_with_player = self.state.get_board_state_as_list(
                    self.game.player)
                self.replay_buffer.add(current_state_with_player, distribution)

                self.state = self.game.perform_action(self.state, action)
                self.game.change_player()

                if self.verbose:
                    self.visualizer.fill_nodes(self.state.get_filled_cells())

                self.ANET.decay_epsilon()

            print(i)

            """ Train ANET """
            root_player = self.game.set_initial_player()
            minibatch = self.replay_buffer.create_minibatch()
            train_states = [case[0] for case in minibatch]
            train_targets = [case[1] for case in minibatch]
            self.ANET.train(train_states, train_targets)

            """ Save model parameters """
            if i+1 % self.save_interval == 0:
                self.ANET.save(i)

            """ Reset game """
            self.mcts.reset(deepcopy(self.initial_state))
            self.state = deepcopy(self.initial_state)
            self.game.player = self.game.set_initial_player()

        self.print_loss_and_accuracy(self.ANET.loss, self.ANET.accuracy)


def main():
    player = StateManager()
    player.play_game()


if __name__ == "__main__":
    main()
