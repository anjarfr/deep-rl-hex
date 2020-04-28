import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class ANET:

    def __init__(self, board_size, dims, lr, activation, optimizer, epsilon, epsilon_decay, epochs, batch_size,
                 save_directory, load_directory):
        self.board_size = board_size
        self.model = NeuralNet(
            k=self.board_size,
            dimensions=dims,
            lr=lr,
            activation=activation,
            optimizer=optimizer
        )
        # --- Parameters for learning ---
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = []
        self.accuracy = []
        self.save_directory = save_directory
        self.load_directory = load_directory

    def generate_tensor(self, input_list):
        tensor = torch.FloatTensor(input_list)
        return tensor

    def train(self, replay_buffer):
        for i in range(self.epochs):
            minibatch = replay_buffer.create_minibatch(self.batch_size)
            train_states = [case[0] for case in minibatch]
            train_targets = [case[1] for case in minibatch]
            state = self.generate_tensor(train_states)
            target = self.generate_tensor(train_targets)
            prediction = self.model(state)
            loss = self.model.update(prediction, target)
        accuracy = self.compute_accuracy(prediction, target)
        self.loss.append(loss)
        self.accuracy.append(accuracy)

    def compute_accuracy(self, prediction, target):
        equal = prediction.argmax(dim=1).eq(target.argmax(dim=1))
        return equal.sum().numpy() / len(prediction)

    def create_legal_indexes(self, legal, moves):
        return [1 if move in legal else 0 for move in moves]

    def re_normalize(self, prediction, legal, moves):
        """ Sets all illegal moves to 0 and renormalizes the 
        distribution """
        legal_indexes = self.create_legal_indexes(legal, moves)
        remove_illegal = [a * b for a,
                          b in zip(prediction.tolist(), legal_indexes)]
        total = sum(remove_illegal)
        if total:
            return [float(i) / total for i in remove_illegal]
        return remove_illegal

    def choose_action(self, state, legal, moves, epsilon, stochastic=False):
        """ Returns chosen action """
        prediction = self.model(self.generate_tensor(state))
        if random.uniform(0, 1) >= epsilon:
            normalized = self.re_normalize(
                prediction, legal, moves)
            if stochastic:
                index = np.random.choice(moves, p=normalized)
            else:
                # Greedy
                index = normalized.index(max(normalized))
            if moves[index] in legal:
                return moves[index]
        return random.choice(legal)

    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay

    def save(self, i):
        torch.save(
            self.model.state_dict(), '{}/ANET_{}_size_{}'.format(
                self.save_directory, i, self.board_size))

    def load(self, i, size):
        self.model.load_state_dict(torch.load(
            '{}/ANET_{}_size_{}'.format(self.load_directory, i, size)))
        self.model.eval()


class NeuralNet(nn.Module):
    """ Neural network """

    def __init__(self, k, dimensions, lr, activation, optimizer):
        super(NeuralNet, self).__init__()

        self.dimensions = dimensions
        self.learning_rate = lr
        self.activation = self.get_activation(activation)

        input_size = 2 * k ** 2 + 2
        output_size = k ** 2

        layers = []

        if len(self.dimensions):
            layers.append(nn.Linear(input_size, self.dimensions[0]))
            layers.append(self.activation) if self.activation else None
            for i in range(len(self.dimensions) - 1):
                layers.append(
                    nn.Linear(self.dimensions[i], self.dimensions[i + 1]))
                layers.append(self.activation) if self.activation else None
            layers.append(nn.Linear(self.dimensions[-1], output_size))
            layers.append(nn.Softmax(dim=-1))
        else:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.Softmax(dim=-1))

        self.model = nn.Sequential(*layers)
        self.model.apply(self.init_weights)

        self.optimizer = self.get_optimizer(
            optimizer, list(self.model.parameters()))

        self.loss_func = nn.BCELoss()

    def update(self, prediction, target):
        """ Update the gradients based on loss """
        self.optimizer.zero_grad()  # Clears gradients
        loss = self.loss_func(prediction, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

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
