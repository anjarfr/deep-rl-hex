from environment.hex import Hex
from environment.visualizer import Visualizer
from agent.anet import ANET
import random
import yaml

random.seed(2020)

with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


class Topp:

    def __init__(self):
        self.size = cfg["game"]["board_size"]
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

        self.p1 = ANET(self.size, dimensions, lr, activation,
                       optimizer, epsilon, epsilon_decay, epochs, batch_size)
        self.p2 = ANET(self.size, dimensions, lr, activation,
                       optimizer, epsilon, epsilon_decay, epochs, batch_size)

    def init_result(self, models):
        self.result = {}
        for i in models:
            self.result[i] = 0

    def round_robin(self):
        step = self.episodes // (self.m - 1)
        models = [i for i in range(0, self.episodes+1, step)]
        self.init_result(models)
        for i in models:
            self.p1.load(i, self.size)
            for j in range(i+step, self.episodes+1, step):
                self.p2.load(j, self.size)
                for _ in range(self.g):
                    self.play_game(i, j)

    def play_game(self, i, j):
        game = Hex(self.size, 1)  # choice([1, 2]))
        state = game.generate_initial_state()
        vis = Visualizer(state, self.size, cfg["display"])
        game.change_player()
        while not game.game_over(state):
            game.change_player()
            legal_actions = game.get_legal_actions(state)
            all_actions = state.get_cell_coord()
            if game.player == 1:
                action = self.p1.choose_action(
                    state.get_board_state_as_list(1),
                    legal_actions,
                    all_actions,
                    0.25, False
                )
                state = game.perform_action(state, action)
            else:
                action = self.p2.choose_action(
                    state.get_board_state_as_list(2),
                    legal_actions,
                    all_actions,
                    0.25, False
                )
                state = game.perform_action(state, action)
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
