import torch
import torch.nn as nn
import torch.optim as optim
import yaml


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
        pass


class NeuralNet(nn.Module):
    """ Neural network """

    def __init__(self, cfg, k):
        super(NeuralNet, self).__init__()

        self.dimensions = cfg["nn"]["dimensions"]
        self.learning_rate = cfg["nn"]["learning_rate"]
        self.activation = self.get_activation(cfg["nn"]["activation_hidden"])

        input_size = 2*k**2 + 2  # 2k**2 + player
        output_size = k**2

        layers = []

        if len(self.dimensions):
            layers.append(nn.Linear(input_size, self.dimensions[0]))
            layers.append(self.activation) if self.activation else None
            for i in range(len(self.dimensions) - 2):
                layers.append(
                    nn.Linear(self.dimensions[i], self.dimensions[i+1]))
                layers.append(self.activation) if self.activation else None
            layers.append(nn.Linear(self.dimensions[-1], output_size))
            layers.append(nn.Softmax(dim=1))
        else:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.Softmax(dim=1))

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


with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

net = ANET(cfg)
print(net.model)
