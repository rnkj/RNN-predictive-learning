import torch
from torch import nn
from torch.nn import functional as F


def lecun_tanh(x):
    return 1.7159 * torch.tanh(2 * x / 3.0)

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self,
                 in_size,
                 hidden_size,
                 out_size):
        super(RecurrentNeuralNetwork, self).__init__()

        self._in_size = in_size
        self._hidden_size = hidden_size
        self._out_size = out_size

        self.i2h = nn.RNNCell(in_size, hidden_size, nonlinearity="tanh")
        self.h2o = nn.Linear(hidden_size, out_size)

    @property
    def in_size(self):
        return self._in_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def out_size(self):
        return self._out_size

    def forward(self, x, h_0=None):
        # x: [N, L, D] dimensions
        # N: batch size
        # L: length of sequences
        # D: dimension of inputs
        hs = []
        ys = []
        h = h_0
        for i in range(x.shape[1]):
            h = self.i2h(x[:, i], h)
            hs.append(h)
            y = lecun_tanh(self.h2o(h))
            ys.append(y)
        hs = torch.stack(hs, dim=1)
        ys = torch.stack(ys, dim=1)
        return hs, ys

class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self,
                 in_size,
                 hidden_size,
                 out_size):
        super(FeedforwardNeuralNetwork, self).__init__()

        self._in_size = in_size
        self._hidden_size = hidden_size
        self._out_size = out_size

        self.i2h = nn.Linear(in_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, out_size)

    @property
    def in_size(self):
        return self._in_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def out_size(self):
        return self._out_size

    def forward(self, x):
        # x: [N, D] dimensions
        # N: batch size
        # D: dimension of inputs
        h = torch.tanh(self.i2h(x))
        y = lecun_tanh(self.h2o(h))
        return h, y
