import torch
import torch.nn as nn
import torch.optim as optim
import random


class ANET:

    def __init__(self, board_size, dims, lr, activation, optimizer, epsilon, epsilon_decay, epochs, batch_size):
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

            #print(prediction)
            #print(target)

            #target = target.argmax(1).to(dtype=torch.int64)
            #print(target)

            loss = self.model.update(prediction, target)


        self.loss.append(loss)

        print("Loss", loss)
        self.accuracy.append(self.compute_accuracy(prediction, target))

    def compute_accuracy(self, prediction, target):
        equal = prediction.argmax(dim=1).eq(target.argmax(dim=1))

        print(equal)
        accuracy = equal.sum().numpy()/len(prediction)

        print("Accuracy", accuracy)
        return accuracy

    def create_legal_indexes(self, moves, legal):
        return [1 if move in legal else 0 for move in moves]

    def re_normalize(self, prediction, legal_indexes):
        """ Sets all illegal moves to 0 and renormalizes the 
        distribution """
        remove_illegal = [a * b for a,
                          b in zip(prediction.tolist(), legal_indexes)]
        total = sum(remove_illegal)
        if total:
            return [float(i) / total for i in remove_illegal]
        return remove_illegal

    def choose_action(self, state, legal_actions, all_actions):
        """ Returns index of chosen action """

        prediction = self.model(self.generate_tensor(state))

        if random.uniform(0, 1) >= self.epsilon:
            legal_indexes = self.create_legal_indexes(
                all_actions, legal_actions)
            normalized = self.re_normalize(prediction, legal_indexes)
            index = normalized.index(max(normalized))
            if all_actions[index] in legal_actions:
                return all_actions[index]
        return random.choice(legal_actions)

    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay

    def save(self, i):
        torch.save(
            self.model.state_dict(), 'new-models/ANET_{}_size_{}'.format(i, self.board_size))

    def load(self, i, size):
        self.model.load_state_dict(torch.load(
            'new-models/ANET_{}_size_{}'.format(i, size)))
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

    def CXE(self, prediction, target):
        return -(target * torch.log(prediction)).sum(dim=1).mean()

    def update(self, prediction, target):
        """ Update the gradients based on loss """
        loss = self.loss_func(prediction, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()  # Clears gradients
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
