import sentiment_NNModel
import sentiment_preprocessing
from sentiment_NNModel import accuracy, sparse_to_Tensor, FFNN, train_model
from sentiment_preprocessing import get_matrix_for_models, load_dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torch import optim


"""load the dataset"""
train_path = '../../data/classification_data/data/sentiment/classification/classification_sentiment_train.jsonl'
test_path = '../../data/classification_data/data/sentiment/classification/classification_sentiment_eval.jsonl'

train_data = load_dataset(train_path)
test_data = load_dataset(test_path)
# hyper_param = {'max_df': 0.6, 'min_df': 0.005}
# convert the dataset to tensor, and split the training data into train and validation set
x_train, y_train, x_test, y_test = get_matrix_for_models(train_data, test_data)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

X_train = sparse_to_Tensor(x_train).to(device)
X_test = sparse_to_Tensor(x_test).to(device)

label_encoder = LabelEncoder()
train_encoded = label_encoder.fit_transform(y_train)
test_encoded = label_encoder.transform(y_test)

Y_train = torch.tensor(train_encoded).to(device)
Y_test = torch.tensor(test_encoded).to(device)


"""train the model with training data"""
input_size = X_train.shape[1]
output_size = 2
# hyper-perameters for model
hidden_sizes = [128, 64, 32]
# hyper-perameters for training
batch_size = 32
learning_rate = 0.01
num_epochs = 1000

model = FFNN(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes)

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_model(X_train, Y_train, model, optimizer, criterion, batch_size, num_epochs)


"""check the accuracy on training and test data"""
Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

accuracy_train = accuracy(Y_pred_train, Y_train)
accuracy_val = accuracy(Y_pred_test, Y_test)

print("Training accuracy", (accuracy_train))
print("Validation accuracy",(accuracy_val))