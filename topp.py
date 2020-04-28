from environment.hex import Hex
from environment.visualizer import Visualizer
from environment.static import generate_board_state, generate_tensor_state
from agent.anet import ANET
import random
import yaml

random.seed(2020)

with open("config.yml", "r", encoding="ISO-8859-1") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


class Topp:

    def __init__(self):
        self.size = cfg["game"]["board_size"]
        self.verbose = cfg["game"]["verbose"]
        self.episodes = cfg['agent']['episodes']
        self.m = cfg['agent']['m']
        self.g = cfg['agent']['g']

        # -- ANET parameters ---
        epsilon_decay = cfg["nn"]["epsilon_decay"]
        dimensions = cfg["nn"]["dimensions"]
        activation = cfg["nn"]["activation_hidden"]
        optimizer = cfg["nn"]["optimizer"]
        epsilon = cfg["nn"]["epsilon"]
        epochs = cfg["nn"]["epochs"]
        lr = cfg["nn"]["learning_rate"]
        batch_size = cfg["nn"]["batch_size"]
        save_directory = cfg["nn"]["save_directory"]
        load_directory = cfg["nn"]["load_directory"]

        self.p1 = ANET(self.size, dimensions, lr, activation,
                       optimizer, epsilon, epsilon_decay, epochs, batch_size, save_directory, load_directory)
        self.p2 = ANET(self.size, dimensions, lr, activation,
                       optimizer, epsilon, epsilon_decay, epochs, batch_size, save_directory, load_directory)

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
            state, self.size), self.size, cfg["display"])
        game.set_player(init_player)
        while not game.game_over(state):
            game.change_player()
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
        # if last_game:
        #     board = generate_board_state(state, self.size)
        #     vis.fill_nodes(board.get_filled_cells())
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
