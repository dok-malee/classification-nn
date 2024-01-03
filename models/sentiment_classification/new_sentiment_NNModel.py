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
from typing import Dict
from nltk import word_tokenize
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class FFNN_with_tfidf(nn.Module):
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

    def forward(self, inputs): #inputsize????
        # input layer to the first hidden layer
        x = self.leakyrelu(self.input_layer(inputs))
        # transition between hidden layers, use LeakyReLU as activation function
        for layer in self.hidden_layers:
            x = self.leakyrelu(layer(x))
        # the last hidden layer to output
        x = self.output_layer(x)

        return x


class FFNN_with_embeddings(nn.Module):
    def __init__(self, hidden_sizes, output_size, embedding_size=100, pretrained_emb=None, vocabulary=None, freeze_embeddings=True):
        """pretrained_emb: word2vec embeddings or None"""
        super().__init__()
        torch.manual_seed(0)
        self.vocabulary = vocabulary
        # input layer to the first hidden layer
        if pretrained_emb != None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=freeze_embeddings)
        else:
            self.embedding = nn.Embedding(len(self.vocabulary), embedding_size)
        self.fc1 = nn.Linear(embedding_size, hidden_sizes[0])
        # multiple hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        # last hidden layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        # input layer to the first hidden layer
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.relu(self.fc1(x))
        # transition between hidden layers, use LeakyReLU as activation function
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        # the last hidden layer to output
        x = self.output_layer(x)
        return x





class FFNN_with_Transformer(nn.Module):
    def __init__(self, tokenizer_model, transformer_model, hidden_sizes, output_size):
        super(FFNN_with_Transformer, self).__init__()
        self.transformer_tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
        self.transformer_model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
        # multiple hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        # last hidden layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.leakyrelu = nn.LeakyReLU()
        for param in self.transformer_model.encoder.layer.parameters():
                param.requires_grad = False

    def forward(self, input):
        # get input_ids and attention_mask for each input
        tokens = self.transformer_tokenizer(input, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # take the [CLS] token of each sequence as the representation of the whole sequence
        x = transformer_output[:, 0, :]
        # hidden layer forward
        for layer in self.hidden_layers:
            x = self.leakyrelu(layer(x))
        # the last hidden layer to output
        x = self.output_layer(x)
        return x



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

def predict(model, inputs, labels):
    # a function that produces the prediction for each example might be helpful
    # to speed things up, you can also use mini-batches here
    predictions = []
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, prediction = torch.max(outputs, 1)
        predictions.extend(prediction.tolist())
    model.train()
    return predictions