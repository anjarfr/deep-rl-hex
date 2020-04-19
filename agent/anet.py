import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import random


class ANET:

    def __init__(self, cfg):
        self.board_size = cfg["game"]["board_size"]
        self.model = NeuralNet(cfg, self.board_size)

    def predict(self):
        return self.model()

    def train(self, state, target):
        prediction = self.model(state)
        self.model.update(prediction, target)

    def re_normalize(self, prediction, legal):
        """ Sets all illegal moves to 0 and renormalizes the 
        distribution """

        remove_illegal = [a*b for a, b in zip(prediction, legal)]   # now illegal actions should be set to 0
        total = sum(remove_illegal)
        normalized = [float(i)/total for i in remove_illegal]

        return normalized

    def choose_action(self, prediction, legal, epsilon):
        """ Returns index of chosen action
        """
        normalized_predictions = self.re_normalize(prediction, legal)

        if random.uniform(0, 1) < epsilon:
            return random.randrange(len(normalized_predictions))

        else:
            return normalized_predictions.index(max(normalized_predictions))


class NeuralNet(nn.Module):
    """ Neural network """

    def __init__(self, cfg, k):
        super(NeuralNet, self).__init__()

        self.dimensions = cfg["nn"]["dimensions"]
        self.learning_rate = cfg["nn"]["learning_rate"]
        self.activation = self.get_activation(cfg["nn"]["activation_hidden"])

        self.optimizers = nn.ModuleDict({
            'adagrad': optim.Adagrad(),
            'adam': optim.Adam(),
            'sgd': optim.SGD(),
            'rmsprop': optim.RMSprop(),
        })

        self.dimensions = cfg["critic"]["dimensions"]
        self.learning_rate = cfg["critic"]["learning_rate"]
        self.activation = cfg["nn"]["activation_hidden"]
        self.optimizer = self.optimizers[cfg["nn"]["optimizer"]]

        input_size = 2*k**2 + 2  # 2k**2 + player
        output_size = k**2

        layers = []

        if len(self.dimensions):
            layers.append(nn.Linear(input_size, self.dimensions[0]))
            layers.append(self.activation) if self.activation else None
            for i in range(len(self.dimensions) - 2):
                layers.append(
                    nn.Linear(self.dimensions[i], self.dimensions[i+1]))
                layers.append(self.activation()) if self.activation else None
            layers.append(nn.Linear(self.dimensions[-1], output_size))
            layers.append(nn.Softmax(dim=1))
        else:
            layers.append(layers.append(nn.Linear(input_size, output_size)))
            layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)
        self.model.apply(init_weights)

        self.optimizer = self.get_optimizer(cfg["nn"]["optimizer"],
                                            list(self.model.parameters()))
        self.loss_func = nn.CrossEntropyLoss()

    def update(self, prediction, target):
        """ Update the gradients based on loss """
        loss = self.loss_func(prediction, target)
        self.optimizer.zero_grad()  # Clears gradients
        loss.backward()
        self.optimizer.step()

    def forward(self, x):
        """ Compute value of state x """
        for layer in self.layers[:-1]:
            x = self.activations[self.activation](layer(x))
        x = torch.softmax(self.layers[-1](x))
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
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
        optimizers = nn.ModuleDict({
            'adagrad': optim.Adagrad(parameters, self.learning_rate),
            'adam': optim.Adam(parameters, self.learning_rate),
            'sgd': optim.SGD(parameters, self.learning_rate),
            'rmsprop': optim.RMSprop(parameters, self.learning_rate),
        })
        return optimizers[optimizer]


with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

net = ANET(cfg)
