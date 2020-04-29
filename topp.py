from environment.hex import Hex
from environment.visualizer import Visualizer
from environment.static import generate_board_state, generate_tensor_state
from agent.anet import ANET
import random
import topp_config as config

random.seed(2020)

class Topp:

    def __init__(self):
        self.size = config.board_size
        self.verbose = config.verbose
        self.episodes = config.episodes
        self.m = config.m
        self.g = config.g

        # -- ANET parameters ---
        dimensions = config.dimensions
        activation = config.activation_hidden
        optimizer = config.optimizer
        lr = config.learning_rate

        self.p1 = ANET(self.size, dimensions, lr, activation, optimizer)
        self.p2 = ANET(self.size, dimensions, lr, activation, optimizer)

    def init_result(self, models):
        self.result = {}
        for i in models:
            self.result[i] = 0

    def round_robin(self):
        step = self.episodes // (self.m - 1)
        models = [i for i in range(0, self.episodes+1, step)]
        init_player = 1
        self.init_result(models)

        for i in models:
            self.p1.load(i, self.size)
            for j in range(i+step, self.episodes+1, step):
                self.p2.load(j, self.size)
                for k in range(self.g):
                    init_player = 3 - init_player
                    self.play_game(i, j, init_player, k == self.g-1)

    def play_game(self, i, j, init_player, last_game):
        game = Hex(self.size, 1)
        state = game.generate_initial_state()
        vis = Visualizer(generate_board_state(
            state, self.size), self.size)
        game.set_player(init_player)
        game_over = game.game_over(state)
        while not game_over:
            legal_actions = game.get_legal_actions(state)
            all_actions = game.get_legal_actions(state)
            if game.player == 1:
                action = self.p1.choose_action(
                    generate_tensor_state(state, 1),
                    legal_actions,
                    all_actions,
                    0, False
                )
                state = game.perform_action(state, action)
            else:
                action = self.p2.choose_action(
                    generate_tensor_state(state, 2),
                    legal_actions,
                    all_actions,
                    0, False
                )
                state = game.perform_action(state, action)
            if game.game_over(state):
                game_over = True
            else:
                game.change_player()
        # if last_game:
        #     board = generate_board_state(state, self.size)
        #     vis.fill_nodes(board.get_filled_cells())
        print(game.game_result())
        if game.game_result() > 0:
            self.result[i] += 1
        else:
            self.result[j] += 1

    def print_result(self):
        for model, result in self.result.items():
            print("{}: {:.1f}%".format(model,
                                       100*result/(self.g * (self.m-1))))


if __name__ == "__main__":
    topp = Topp()
    topp.round_robin()
    topp.print_result()
