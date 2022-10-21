import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from model import FeedforwardNeuralNetwork
from dataset import SingleCircleDataset


HIDDEN_SIZE = 64
EPOCHS = 10000
LEARNING_RATE = 0.001

STOP = 2 * np.pi
LENGTH = 50

def closed_loop_prediction(fnn: FeedforwardNeuralNetwork,
                           test_loader: SingleCircleDataset):
    """
    Closed-loop prediction for `test_loader` input.
    The network `fnn` always receives inputs from `test_loader`.
    """
    fnn.eval()
    inputs, teachers = test_loader.get_sequence()
    outputs = []
    for i in range(inputs.shape[1]):
        h, y = fnn(inputs[:, i])
        outputs.append(y)
    outputs = torch.stack(outputs, dim=1)
    return outputs.detach().numpy()

def open_loop_prediction(fnn: FeedforwardNeuralNetwork,
                         test_loader: SingleCircleDataset,
                         prediction_steps=50):
    """
    Open-loop prediction for `prediction_steps` steps.
    The network `fnn` receives only the first four data points of `test_loader`.
    """
    fnn.eval()
    inputs, teachers = test_loader.get_sequence()
    y = None
    outputs = []
    for step_i in range(prediction_steps):
        if step_i < 4:
            x = inputs[:, step_i]
        else:
            x = y
        h, y = fnn(x)
        outputs.append(y)
    outputs = torch.stack(outputs, dim=1)
    return outputs.detach().numpy()

def train_one_step(fnn: FeedforwardNeuralNetwork,
                   optimizer: torch.optim.Optimizer,
                   train_loader: SingleCircleDataset):
    """
    A single training process - forward, backward and parameter update
    """
    fnn.train()
    inputs, teachers = train_loader.get_sequence()
    optimizer.zero_grad()
    outputs = []
    for i in range(inputs.shape[1]):
        h, y = fnn(inputs[:, i])
        outputs.append(y)
    outputs = torch.stack(outputs, dim=1)
    loss = F.mse_loss(outputs, teachers)
    loss.backward()
    optimizer.step()
    return loss.data

def train_fnn(fnn: FeedforwardNeuralNetwork,
              train_loader: SingleCircleDataset,
              epochs: int,
              learning_rate: float,
              use_cuda=True):
    """
    `epochs` repetitions of training processes
    """
    if torch.cuda.is_available() and use_cuda:
        fnn.cuda()
    optimizer = optim.Adam(params=fnn.parameters(), lr=learning_rate)

    losses = []
    for epoch_i in tqdm(range(epochs)):
        losses.append(train_one_step(fnn, optimizer, train_loader))
    return losses

if __name__ == "__main__":
    train_loader = SingleCircleDataset(stop=STOP, length=LENGTH)
    fnn = FeedforwardNeuralNetwork(train_loader.dims, HIDDEN_SIZE, train_loader.dims)

    losses = train_fnn(fnn, train_loader, EPOCHS, LEARNING_RATE, use_cuda=True)

    teachers = train_loader.teachers.detach().numpy()
    closed_preds = closed_loop_prediction(fnn, train_loader)
    num_cycles = 3
    open_preds = open_loop_prediction(fnn, train_loader, prediction_steps=num_cycles * LENGTH)

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
    ax.set(xlim=(-0.5, 4.5), ylim=(-0.5, 4.5))
    ax.set_title("Closed-loop")
    # open-loop
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(teachers[0, :, 0], teachers[0, :, 1], c="k")
    ax.scatter(open_preds[0, :, 0], open_preds[0, :, 1], s=20)
    ax.axis("equal")
    ax.set(xlim=(-0.5, 4.5), ylim=(-0.5, 4.5))
    ax.set_title("Open-loop ({} cycles)".format(num_cycles))
    plt.show()
