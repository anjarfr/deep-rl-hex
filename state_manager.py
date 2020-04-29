import config
import matplotlib.pyplot as plt

from agent.anet import ANET
from agent.buffer import ReplayBuffer
from agent.mcts import MCTS
from environment.hex import Hex
from environment.visualizer import Visualizer
from environment.static import generate_board_state, generate_tensor_state


class StateManager:
    def __init__(self):
        """ initialize all the shizz """

        self.verbose = config.verbose
        self.episodes = config.episodes
        self.simulations = config.simulations
        self.display_last_game = config.display_last_game
        self.save_interval = self.episodes // (config.m)

        # -- Hex game and sim game initialization --
        self.size = config.board_size
        initial_player = config.player
        self.game = Hex(self.size, initial_player)
        self.initial_state = self.game.generate_initial_state()
        self.sim_game = Hex(self.size, initial_player)
        self.sim_game_state = self.sim_game.generate_initial_state()
        self.state = [*self.initial_state]

        # -- ANET parameters ---
        epsilon_decay = config.epsilon_decay
        dimensions = config.dimensions
        activation = config.activation_hidden
        optimizer = config.optimizer
        epsilon = config.epsilon
        epochs = config.epochs
        lr = config.learning_rate
        batch_size = config.batch_size
        max_buffer_length = config.max_buffer_length
        save_directory = config.save_directory
        load_directory = config.load_directory

        self.ANET = ANET(
            self.size,
            dimensions,
            lr,
            activation,
            optimizer,
            epsilon,
            epsilon_decay,
            epochs,
            batch_size,
            save_directory,
            load_directory,
        )
        self.mcts = MCTS(
            self.sim_game, self.sim_game_state, self.simulations, self.ANET
        )

        init_board = generate_board_state(self.initial_state, self.size)
        self.visualizer = Visualizer(init_board, self.size)
        self.replay_buffer = ReplayBuffer(max_buffer_length)

    def print_loss_and_accuracy(self, loss, accuracy):
        plt.plot(loss)
        plt.ylabel("Loss")
        plt.plot(accuracy)
        plt.ylabel("Accuracy")
        plt.xlabel("Iteration")
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
                tensor_state = generate_tensor_state(
                    self.state, self.game.player
                )
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
                # self.print_loss_and_accuracy(self.ANET.loss,
                #                              self.ANET.accuracy)
            if i + 1 == self.episodes:
                self.ANET.save(i + 1)

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
