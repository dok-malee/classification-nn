import sentiment_NNModel
import sentiment_preprocessing
from sentiment_NNModel import accuracy, FFNN, train_model
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
# X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2)

"""convert the sparse matrix and label lists into pytorch tensor"""
def sparse_to_Tensor(sparse_features):
    sparse_tensor = torch.sparse_coo_tensor(
        torch.LongTensor(np.vstack(sparse_features.nonzero())),
        torch.FloatTensor(sparse_features.data),
        torch.Size(sparse_features.shape),
    )
    return sparse_tensor

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

X_train = sparse_to_Tensor(x_train).to(device)
X_test = sparse_to_Tensor(x_test).to(device)

label_encoder = LabelEncoder()
train_encoded = label_encoder.fit_transform(y_train)
test_encoded = label_encoder.transform(y_test)

Y_train = torch.tensor(train_encoded).to(device)
Y_test = torch.tensor(test_encoded).to(device)


"""train the model with training data"""
# hyper-perameters
hidden_sizes = [128, 64, 32]
batch_size = 32
learning_rate = 0.001
num_epochs = 10
# parameters
input_size = X_train.shape[1]
output_size = 2

fn = FFNN(input_size, hidden_sizes, output_size)

fn.to('cuda')
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(fn.parameters(), lr=learning_rate)

print('Final loss', train_model(X_train, Y_train, fn, opt, loss_fn, num_epochs))



"""check the accuracy on training and test data"""
Y_pred_train = fn.predict(X_train)
Y_pred_test = fn.predict(X_test)

accuracy_train = accuracy(Y_pred_train, Y_train)
accuracy_val = accuracy(Y_pred_test, Y_test)

print("Training accuracy", (accuracy_train))
print("Validation accuracy",(accuracy_val))