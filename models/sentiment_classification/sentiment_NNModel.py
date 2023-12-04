import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score



print("GPU Available:", torch.cuda.is_available())



def accuracy(y_hat, y):
    """function to calculate accuracy"""
    pred = torch.argmax(y_hat, dim=1)
    return (pred == y).float().mean()


class FFNN(nn.Module):
    """construct Feed Forward Neural Network"""
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        torch.manual_seed(0)
        # nn.Linear(input_size=tfidf_tensor.shape[1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # input layer to the first hidden layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        # multiple hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)])

        # last hidden layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # input layer to the first hidden layer
        x = torch.relu(self.input_layer(x))

        # transition between hidden layers
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        # the last hidden layer to output
        x = self.output_layer(x)

        return x

    def predict(self, X):
        Y_pred = self.forward(X)
        print(Y_pred)
        return Y_pred


def train_model(x, y, model, opt, loss_fn, batch_size=32, num_epochs=1000):
    """train model with batch, but I think there are some bugs with the loss function part."""
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_step = len(dataloader)
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            # Forward pass
            logits = model(batch_x)

            # Compute the loss
            loss = loss_fn(logits, batch_y)

            # Backward pass and optimization
            loss.backward()
            opt.step()
            opt.zero_grad()
            # print('optimized')

            total_loss += loss.item()

        # Average loss for the epoch
        average_loss = total_loss / len(dataloader)
        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, epoch + 1, total_step, average_loss))

    return average_loss






