import yaml
import matplotlib.pyplot as plt

from agent.anet import ANET
from agent.buffer import ReplayBuffer
from agent.mcts import MCTS
from environment.hex import Hex
from environment.visualizer import Visualizer
from environment.static import generate_board_state, generate_tensor_state

with open("config.yml", "r", encoding="ISO-8859-1") as ymlfile:
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
        self.save_interval = self.episodes // (cfg["agent"]["m"]-1)

        # -- Hex game and sim game initialization --
        self.size = cfg["game"]["board_size"]
        initial_player = cfg["game"]["player"]
        self.game = Hex(self.size, initial_player)
        self.initial_state = self.game.generate_initial_state()
        self.sim_game = Hex(self.size, initial_player)
        self.sim_game_state = self.sim_game.generate_initial_state()
        self.state = [*self.initial_state]

        # -- ANET parameters ---
        epsilon_decay = cfg["nn"]["epsilon_decay"]
        dimensions = cfg["nn"]["dimensions"]
        activation = cfg["nn"]["activation_hidden"]
        optimizer = cfg["nn"]["optimizer"]
        epsilon = cfg["nn"]["epsilon"]
        epochs = cfg["nn"]["epochs"]
        lr = cfg["nn"]["learning_rate"]
        batch_size = cfg["nn"]["batch_size"]
        max_buffer_length = cfg["nn"]["max_buffer_length"]
        save_directory = cfg["nn"]["save_directory"]
        load_directory = cfg["nn"]["load_directory"]

        self.ANET = ANET(self.size, dimensions, lr, activation,
                         optimizer, epsilon, epsilon_decay, epochs, batch_size, save_directory, load_directory)
        self.mcts = MCTS(cfg, self.sim_game, self.sim_game_state,
                         self.simulations, self.ANET)

        init_board = generate_board_state(self.initial_state, self.size)
        self.visualizer = Visualizer(
            init_board, self.size, cfg["display"])
        self.replay_buffer = ReplayBuffer(max_buffer_length)

    def print_loss_and_accuracy(self, loss, accuracy):
        plt.plot(loss)
        plt.ylabel('Loss')
        plt.plot(accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('Iteration')
        plt.show()

    def print_game(self, state):
        board = generate_board_state(state, self.size)
        self.visualizer.fill_nodes(board.get_filled_cells())

    def play_game(self):
        """ One complete game """
        for i in range(self.episodes):
            self.game.set_initial_player()

            while not self.game.game_over(self.state):
                """ Do simulations and choose best action """
                distribution, action = self.mcts.uct_search(self.game.player)
                tensor_state = generate_tensor_state(self.state,
                                                     self.game.player)
                self.replay_buffer.add(tensor_state, distribution)

                self.state = self.game.perform_action(self.state, action)
                self.game.change_player()

                self.mcts.reset([*self.state])

            if self.verbose:
                self.print_game(self.state)

            print(i, self.ANET.epsilon)

            """ Train ANET """

            self.ANET.train(self.replay_buffer)
            self.ANET.decay_epsilon()

            """ Save model parameters """
            if i % self.save_interval == 0:
                self.ANET.save(i)
            if i+1 == self.episodes:
                self.ANET.save(i+1)

            """ Reset game """
            self.mcts.reset([*self.initial_state])
            self.state = [*self.initial_state]
            self.game.player = self.game.set_initial_player()

        self.print_loss_and_accuracy(self.ANET.loss, self.ANET.accuracy)


def main():
    player = StateManager()
    player.play_game()


if __name__ == "__main__":
    main()
