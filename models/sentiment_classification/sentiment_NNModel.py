import numpy as np
import torch
import wandb
import math
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict
from nltk import word_tokenize
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns


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

    def forward(self, inputs): #inputsize????
        # input layer to the first hidden layer
        x = self.leakyrelu(self.input_layer(inputs))
        # transition between hidden layers, use LeakyReLU as activation function
        for layer in self.hidden_layers:
            x = self.leakyrelu(layer(x))
        # the last hidden layer to output
        x = self.output_layer(x)

        return x




def train_model(x, y,model, opt, loss_fn, batch_size=32, epochs = 1000, print_epoch=100):
    best_model = None
    best_dev_f1 = 0.0

    dataset = torch.utils.data.TensorDataset(x, y)
    k_folds = 5
    # Initialize KFold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    # Iterate over folds
    for fold, (train_idx, dev_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}:")
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        dev_dataset = torch.utils.data.Subset(dataset, dev_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_x, dev_y = map(torch.stack, zip(*dev_dataset))
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
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
            average_loss = total_loss / len(train_loader)

            dev_predictions = predict(model, dev_x)
            dev_f1 = sklearn.metrics.f1_score(dev_y, dev_predictions, average="weighted")
            dev_acc = sklearn.metrics.accuracy_score(dev_y, dev_predictions)

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_model = model.state_dict()  # store the model with highest f1score on val data
            wandb.log({"loss": average_loss, "dev acc": dev_acc, "dev_f1": dev_f1, "dev_acc": dev_acc, "best_dev_f1": best_dev_f1})
            if (epoch + 1) % print_epoch == 0:
                print(f"Fold:{fold+1}/{k_folds}\t Epoch: {epoch + 1}/{epochs}\t Average Loss: {average_loss}\t Dev F1 Score: {dev_f1}\t Dev Acc Score: {dev_acc}")
    print("Best Dev F1 Score:", best_dev_f1)
    model.load_state_dict(best_model)


def predict(model, inputs):
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

def get_classification_report(y_true, y_pred, label_names):
    report = classification_report(y_true, y_pred, target_names=label_names)
    return report

def show_confusion_matrix(y_true, y_pred, label_names, model_name):
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the plot to a file (e.g., PNG)
    plt.savefig(f'{model_name}_confusion_matrix.png')

    # Show the plot (optional)
    plt.show()


# class FFNN_with_embeddings(nn.Module):
#     def __init__(self, hidden_sizes, output_size, embedding_size=100, pretrained_emb=None, vocabulary=None, freeze_embeddings=True):
#         """pretrained_emb: word2vec embeddings or None"""
#         super().__init__()
#         torch.manual_seed(0)
#         self.vocabulary = vocabulary
#         # input layer to the first hidden layer
#         if pretrained_emb != None:
#             self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=freeze_embeddings)
#         else:
#             self.embedding = nn.Embedding(len(self.vocabulary), embedding_size)
#         self.fc1 = nn.Linear(embedding_size, hidden_sizes[0])
#         # multiple hidden layers
#         self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
#         # last hidden layer
#         self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
#         self.relu = nn.ReLU()
#
#
#     def forward(self, x):
#         # input layer to the first hidden layer
#         x = self.embedding(x)
#         x = x.mean(dim=1)
#         x = self.relu(self.fc1(x))
#         # transition between hidden layers, use LeakyReLU as activation function
#         for layer in self.hidden_layers:
#             x = self.relu(layer(x))
#         # the last hidden layer to output
#         x = self.output_layer(x)
#         return x


# miniLM_tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
# miniLM_model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")


# class FFNN_with_Transformer(nn.Module):
#     def __init__(self, hidden_sizes, output_size, tokenizer_model=miniLM_tokenizer, transformer_model=miniLM_model):
#         super(FFNN_with_Transformer, self).__init__()
#         self.transformer_tokenizer = tokenizer_model
#         self.transformer_model = transformer_model
#         self.fc1 = nn.Linear(384, hidden_sizes[0])
#         # multiple hidden layers
#         self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
#         # last hidden layer
#         self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
#         self.relu = nn.ReLU()
#         for param in self.transformer_model.encoder.layer.parameters():
#                 param.requires_grad = False
#
#     def forward(self, input_ids, attention_mask):
#         # get input_ids and attention_mask for each input
#         # tokens = self.transformer_tokenizer(input, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
#         # input_ids = tokens["input_ids"]
#         # attention_mask = tokens["attention_mask"]
#         transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
#
#         # take the [CLS] token of each sequence as the representation of the whole sequence
#         x = transformer_output[:, 0, :]
#         # hidden layer forward
#         x = self.relu(self.fc1(x))
#         for layer in self.hidden_layers:
#             x = self.relu(layer(x))
#         # the last hidden layer to output
#         x = self.output_layer(x)
#         return x


# def get_batches_for_transformer(texts, labels, batch_size, shuffle=True):
#     n = len(texts)
#     indices = [i for i in range(n)]
#     if shuffle:
#         indices = sklearn.utils.shuffle(indices)
#
#     for start_i in range(0, n, batch_size):
#         # get batch_texts, batch_labels
#         end_i = min(n, start_i + batch_size)
#         batch_indices = indices[start_i: end_i]
#         batch_texts = [texts[i] for i in batch_indices]
#         batch_labels = torch.tensor([labels[i] for i in batch_indices])
#         batch_inputs = miniLM_tokenizer(batch_texts, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
#         batch_input_ids = batch_inputs["input_ids"]
#         batch_attention_mask = batch_inputs["attention_mask"]
#         yield batch_input_ids, batch_attention_mask, batch_labels

# def train_model_transformer(x, y, val_x, val_y, model, opt, loss_fn, batch_size=32, epochs = 1000): # x: list of strings
#     best_model = None
#     best_val_f1 = 0.0
#     for epoch in range(epochs):
#         total_loss = 0.0
#         train_input = list(get_batches_for_transformer(x, y, batch_size=batch_size))
#         batch_num = len(train_input)
#         for batch in train_input:
#             opt.zero_grad()
#             input_ids, attention_mask, labels = batch
#             # print("Inputs shape: ",inputs.shape)
#             outputs = model(input_ids, attention_mask)
#             # print("outputs.shape", outputs.shape)
#             loss = loss_fn(outputs, labels)
#             loss.backward()
#             opt.step()
#
#             total_loss += loss.item()
#         average_loss = total_loss / batch_num
#
#         val_predictions = predict(model, val_x)
#         val_f1 = sklearn.metrics.f1_score(val_y, val_predictions, average="weighted")
#         val_acc = sklearn.metrics.accuracy_score(val_y, val_predictions)
#
#         if val_f1 > best_val_f1:
#             best_val_f1 = val_f1
#             best_model = model.state_dict()  # store the model with highest f1score on val data
#
#         if (epoch + 1) % 1 == 0:
#             print(f"Epoch: {epoch + 1}/{epochs}\t Average Loss: {average_loss}\t Validation Acc: {val_acc}\t Validation F1 Score: {val_f1}\t Best Validation F1 Score: {best_val_f1}")



# def predict_transformer(model, texts, labels, batch_size, shuffle=True):
#     model.eval()
#     batches = list(get_batches_for_transformer(texts, labels, batch_size=batch_size, shuffle=shuffle))
#     predictions = []
#     with torch.no_grad():
#         for batch in batches:
#             input_ids, mask_attention, labels = batch
#             outputs = model(input_ids, mask_attention)
#             _, prediction = torch.max(outputs, 1)
#             predictions.extend(prediction.tolist())
#     model.train()
#     return predictions

