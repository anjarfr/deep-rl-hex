import torch
import torch.nn as nn
import torch.optim as optim


class ANET:

    def __init__(self, cfg):
        self.board_size = cfg[]
        self.model = NeuralNet(cfg, board_size)
    
    def predict(self):
        return self.model()

    def train(self, target):
        prediction = self.model()
        self.model.update(prediction, target)


class NeuralNet(nn.Module):
    """ Neural network """

    def __init__(self, cfg, k):
        super(TorchNet, self).__init__()

        self.loss_func = nn.CrossEntropyLoss()

        self.activations = nn.ModuleDict({
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'linear': nn.Linear(),
            'relu': nn.ReLU(),
        })

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

        input_size = 2*k**2 + 2  # 2k² + player
        output_size = k**2

        layers = []
        layers.append(nn.Linear(input_size, self.dimensions[0]))

        for i in range(len(self.dimensions) - 2):
            layers.append(nn.Linear(self.dimensions[i], self.dimensions[i+1]))

        layers.append(nn.Linear(self.dimensions[-1], output_size))

        self.layers = nn.ModuleList(layers)

    def update(self, prediction, target):
        """ Update the gradients based on loss """
        loss = self.loss_func(prediction, target)
        self.optimizer.zero_grad()  # Clears gradients
        loss.backward()
        optimizer.step()

    def forward(self, x):
        """ Compute value of state x """
        for layer in self.layers[:-1]:
            x = self.activations[self.activation](layer(x))
        x = torch.softmax(self.layers[-1](x))
        return x
