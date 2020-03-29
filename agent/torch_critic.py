import torch
import torch.nn as nn
from .critic import Critic

torch.manual_seed(42)


class NeuralNetCritic(Critic):

    def __init__(self, cfg, init_state):
        super(NeuralNetCritic, self).__init__(cfg)
        self.model = TorchNet(cfg, init_state)

    def calculate_TD_error(self, state, succ_state, reward):
        """ Calculates TD error """
        state = self.generate_state(state)
        succ_state = self.generate_state(succ_state)

        value_function_state = self.model.forward(state)
        value_function_succ_state = self.model.forward(succ_state)

        TD_error = reward + self.discount_factor * \
            value_function_succ_state - value_function_state
        return TD_error[0].item()

    def reset_eligibility(self):
        """ Reset all eligibility traces to 0 """
        for i, layer in enumerate(self.model.eligibility):
            self.model.eligibility[i] = torch.zeros(layer.shape)

    def generate_state(self, state):
        """ Creates a tensor of state """
        listed_state = []
        for c in state:
            listed_state.append(int(c))
        tensor = torch.tensor(listed_state, dtype=torch.float32)
        return tensor

    def update_value_function(self, state, TD_error, reward=0, succ_state=None):
        """ Update gradients of neural net to incorporate eligibility traces """
        state = self.generate_state(state)
        succ_state = self.generate_state(succ_state)

        prediction = self.model.forward(state)
        target = reward + self.discount_factor * self.model.forward(succ_state)

        self.model.update(prediction, target)


class TorchNet(nn.Module):
    """ Neural network """

    def __init__(self, cfg, init_state):
        super(TorchNet, self).__init__()

        self.loss_func = nn.MSELoss()

        self.dimensions = cfg["critic"]["dimensions"]
        self.learning_rate = cfg["critic"]["learning_rate"]
        self.eligibility_decay = cfg["critic"]["eligibility_decay"]
        self.discount_factor = cfg["critic"]["discount_factor"]

        self.eligibility = []

        layers = []

        layers.append(nn.Linear(len(init_state), self.dimensions[0]))
        for i in range(len(self.dimensions) - 1):
            layers.append(nn.Linear(self.dimensions[i], self.dimensions[i+1]))
        self.layers = nn.ModuleList(layers)

        self.initialize_eligibility()

    def initialize_eligibility(self):
        """ Initialize eligibilities of weights and biases in the network to 0 """
        for params in self.parameters():
            self.eligibility.append(torch.zeros(params.shape))

    def update(self, prediction, target):
        """ Modify the gradients in the neural network to incorporate eligibility traces """
        loss = self.loss_func(prediction, target)
        self.zero_grad()  # Clears gradients
        loss.backward(retain_graph=True)

        with torch.no_grad():
            for i, params in enumerate(self.parameters()):
                self.eligibility[i] = self.discount_factor * \
                    self.eligibility_decay * \
                    self.eligibility[i] + params.grad.data
                params = params + self.learning_rate * \
                    (target - prediction) * self.eligibility[i]

    def forward(self, x):
        """ Compute value of state x """
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x
