import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import random


class ANET:

    def __init__(self, cfg):
        self.board_size = cfg["game"]["board_size"]
        self.epsilon = cfg["nn"]["epsilon"]
        self.epsilon_decay = cfg["nn"]["epsilon_decay"]
        self.model = NeuralNet(cfg, self.board_size)

    def generate_tensor(self, state, player):
        """ Creates a tensor of state where state is of class Board """
        if player == 1:
            listed_state = [0, 1]
        else:
            listed_state = [1, 0]
        for row in state.cells:
            for cell in row:
                listed_state.extend(list(cell.coordinates))
        tensor = torch.tensor(listed_state, dtype=torch.float32)
        return tensor

    def predict(self, state, player):
        return self.model(state)

    def train(self, state: list, target: list, player: int):
        state = self.generate_tensor(state, player)
        predictions = self.model(state)
        self.model.update(predictions, targets)

    def create_legal_indexes(self, moves, legal):
        return [1 if move in legal else 0 for move in moves]

    def re_normalize(self, prediction, legal_indexes):
        """ Sets all illegal moves to 0 and renormalizes the 
        distribution """
        remove_illegal = [a*b for a,
                          b in zip(prediction.tolist(), legal_indexes)]
        total = sum(remove_illegal)
        normalized = [float(i)/total for i in remove_illegal]
        return normalized

    def choose_action(self, state, player, legal):
        """ Returns index of chosen action
        """
        # TODO Must be called from somewhere, maybe in MCTS after prediction has been made in ANET, or in ANET
        prediction = self.model(self.generate_tensor(state, player))
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(legal)
        else:
            moves = state.get_cell_coord()
            legal_indexes = self.create_legal_indexes(moves, legal)
            normalized = self.re_normalize(prediction, legal_indexes)
            index = normalized.index(max(normalized))
            return moves[index]

    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay

    def save(self, i):
        torch.save(
            self.model.state_dict(), 'models/ANET_{}_size_{}.pth'.format(i, self.board_size))

    def load(self, i):
        self.model.load_state_dict(torch.load(
            'models/ANET_{}_size_{}.pth'.format(i, self.board_size)))
        self.model.eval()


class NeuralNet(nn.Module):
    """ Neural network """

    def __init__(self, cfg, k):
        super(NeuralNet, self).__init__()

        self.dimensions = cfg["nn"]["dimensions"]
        self.learning_rate = cfg["nn"]["learning_rate"]
        self.activation = self.get_activation(cfg["nn"]["activation_hidden"])

        input_size = 2*k**2 + 2
        output_size = k**2

        layers = []

        if len(self.dimensions):
            layers.append(nn.Linear(input_size, self.dimensions[0]))
            layers.append(self.activation) if self.activation else None
            for i in range(len(self.dimensions) - 1):
                layers.append(
                    nn.Linear(self.dimensions[i], self.dimensions[i+1]))
                layers.append(self.activation) if self.activation else None
            layers.append(nn.Linear(self.dimensions[-1], output_size))
            layers.append(nn.Softmax(dim=-1))
        else:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.Softmax(dim=-1))

        self.model = nn.Sequential(*layers)
        self.model.apply(self.init_weights)

        self.optimizer = self.get_optimizer(
            cfg["nn"]["optimizer"], list(self.model.parameters()))
        self.loss_func = nn.CrossEntropyLoss()

    def update(self, prediction, target):
        """ Update the gradients based on loss """
        loss = self.loss_func(prediction, target)
        self.optimizer.zero_grad()  # Clears gradients
        loss.backward()
        self.optimizer.step()

    def forward(self, x):
        return self.model(x)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_activation(self, activation):
        activations = nn.ModuleDict({
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'linear': None,
            'relu': nn.ReLU(),
        })
        return activations[activation]

    def get_optimizer(self, optimizer, parameters):
        optimizers = {
            'adagrad': optim.Adagrad(parameters, self.learning_rate),
            'adam': optim.Adam(parameters, self.learning_rate),
            'sgd': optim.SGD(parameters, self.learning_rate),
            'rmsprop': optim.RMSprop(parameters, self.learning_rate),
        }
        return optimizers[optimizer]
