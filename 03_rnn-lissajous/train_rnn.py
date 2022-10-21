import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from model import RecurrentNeuralNetwork
from dataset import SingleLissajousDataset


HIDDEN_SIZE = 64
EPOCHS = 10000
LEARNING_RATE = 0.001

STOP = 2 * np.pi
LENGTH = 50

def closed_loop_prediction(rnn: RecurrentNeuralNetwork,
                           test_loader: SingleLissajousDataset):
    """
    Closed-loop prediction for `test_loader` input.
    The network `rnn` always receives inputs from `test_loader`.
    """
    rnn.eval()
    inputs, teachers = test_loader.get_sequence()
    hs, outputs = rnn(inputs)
    return outputs.detach().numpy()

def open_loop_prediction(rnn: RecurrentNeuralNetwork,
                         test_loader: SingleLissajousDataset,
                         prediction_steps=50):
    """
    Open-loop prediction for `prediction_steps` steps.
    The network `rnn` receives only the first data point of `test_loader`.
    """
    rnn.eval()
    inputs, teachers = test_loader.get_sequence()
    x = inputs[:, 0:1] # shape: (N, 1, D)
    h = None
    outputs = []
    for _ in range(prediction_steps):
        h, y = rnn(x, h_0=h)
        outputs.append(y)
        x = y
        h = h[:, -1] # shape (N, 1, D_h) -> (N, D_h)
    outputs = torch.concat(outputs, dim=1) # shape: (N, L, D)
    return outputs.detach().numpy()

def train_one_step(rnn: RecurrentNeuralNetwork,
                   optimizer: torch.optim.Optimizer,
                   train_loader: SingleLissajousDataset):
    """
    A single training process - forward, backward and parameter update
    """
    rnn.train()
    inputs, teachers = train_loader.get_sequence()
    optimizer.zero_grad()
    hs, outputs = rnn(inputs)
    loss = F.mse_loss(outputs, teachers)
    loss.backward()
    optimizer.step()
    return loss.data

def train_rnn(rnn: RecurrentNeuralNetwork,
              train_loader: SingleLissajousDataset,
              epochs: int,
              learning_rate: float,
              use_cuda=True):
    """
    `epochs` repetitions of training processes
    """
    if torch.cuda.is_available() and use_cuda:
        rnn.cuda()
    optimizer = optim.Adam(params=rnn.parameters(), lr=learning_rate)

    losses = []
    for epoch_i in tqdm(range(epochs)):
        losses.append(train_one_step(rnn, optimizer, train_loader))
    return losses

if __name__ == "__main__":
    train_loader = SingleLissajousDataset(stop=STOP, length=LENGTH)
    rnn = RecurrentNeuralNetwork(train_loader.dims, HIDDEN_SIZE, train_loader.dims)

    losses = train_rnn(rnn, train_loader, EPOCHS, LEARNING_RATE, use_cuda=True)

    teachers = train_loader.teachers.detach().numpy()
    closed_preds = closed_loop_prediction(rnn, train_loader)
    num_cycles = 3
    open_preds = open_loop_prediction(rnn, train_loader, prediction_steps=num_cycles * LENGTH)

    # plot predictions
    fig = plt.figure()
    gs = GridSpec(2, 2)
    # loss
    ax = fig.add_subplot(gs[0, :])
    ax.plot(losses)
    ax.set_yscale("log")
    ax.set_title("training error")
    # closed-loop
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(teachers[0, :, 0], teachers[0, :, 1], c="k")
    ax.scatter(closed_preds[0, :, 0], closed_preds[0, :, 1], s=20)
    ax.axis("equal")
    ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    ax.set_title("Closed-loop")
    # open-loop
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(teachers[0, :, 0], teachers[0, :, 1], c="k")
    ax.scatter(open_preds[0, :, 0], open_preds[0, :, 1], s=20)
    ax.axis("equal")
    ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    ax.set_title("Open-loop ({} cycles)".format(num_cycles))
    plt.show()
