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


def sparse_to_Tensor(sparse_features):
    """convert the sparse matrix and label lists into pytorch tensor"""
    sparse_tensor = torch.sparse_coo_tensor(
        torch.LongTensor(np.vstack(sparse_features.nonzero())),
        torch.FloatTensor(sparse_features.data),
        torch.Size(sparse_features.shape),
    )
    return sparse_tensor


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        torch.manual_seed(0)

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        # input layer to the first hidden layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        # multiple hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        # last hidden layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        # input layer to the first hidden layer
        x = self.leakyrelu(self.input_layer(x))
        # transition between hidden layers, use LeakyReLU as activation function
        for layer in self.hidden_layers:
            x = self.leakyrelu(layer(x))
        # the last hidden layer to output
        x = self.output_layer(x)

        return x

    def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred


def train_model(x, y, model, opt, loss_fn, batch_size=32, epochs = 1000):
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            # Clear gradients of parameters
            opt.zero_grad()
            # Forward pass
            outputs = model(batch_x)
            # Calculate Loss: softmax --> cross entropy loss
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            # Updating parameters
            opt.step()
            total_loss += loss.item()
        # Average loss for the epoch
        average_loss = total_loss / len(dataloader)
        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, average_loss))






