import torch
from torch import nn
from torch.nn import functional as F


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
        h = F.relu(self.i2h(x))
        y = F.relu(self.h2o(h))
        return h, y
