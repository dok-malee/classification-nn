from sklearn.preprocessing import LabelEncoder
from letter_preprocessing import load_dataset, create_sparse_vectors, sparse_to_Tensor
from author_classification_model import train_model, evaluate_model, show_confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, precision_score, recall_score, f1_score

# Define the Feedforward Neural Network model
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.batch_norm = nn.BatchNorm1d(hidden_size1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

 # paths to files
path_to_train_file = "/Users/sarahannauffelmann/desktop/01_WiSe23:24/Seminar Klassifizierung/Projekt FFNN/letters/classifier_data_train.jsonl"
path_to_test_file = "/Users/sarahannauffelmann/desktop/01_WiSe23:24/Seminar Klassifizierung/Projekt FFNN/letters/classifier_data_eval.jsonl"
    
# training and test data
train_inst, train_author_labels, train_lang_labels = load_dataset(path_to_train_file)
test_inst, test_author_labels, test_lang_labels = load_dataset(path_to_test_file)

# sparse vectors to tensors
x_train_sparse, x_test_sparse = create_sparse_vectors(train_inst, test_inst, 0.5, 5)

X_train = sparse_to_Tensor(x_train_sparse)
X_test = sparse_to_Tensor(x_test_sparse)

# labels to vector encoding
label_enc = LabelEncoder()
labels_encoded_train = label_enc.fit_transform(train_lang_labels)
labels_encoded_test = label_enc.transform(test_lang_labels)
target_names = label_enc.classes_   
#print(target_names)

y_train = torch.tensor(labels_encoded_train)
y_test = torch.tensor(labels_encoded_test)

# create Dataloader
# https://machinelearningmastery.com/training-a-pytorch-model-with-dataloader-and-dataset/
dataloader_train = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=64)
dataloader_test = DataLoader(list(zip(X_test, y_test)), shuffle=True, batch_size=64)
#for X_batch, y_batch in dataloader_train:
    #print(X_batch, y_batch)

# model parameters
input_size = X_train.shape[1]
output_size = 7         #  7 different authors
hidden1 = 512
hidden2 = 512

model = FFNN(input_size, hidden1, hidden2, output_size)
model_name = "FFNN_sparse"

# training parameters
num_epochs = 2
learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
# for epoch in range(num_epochs):
#      train_loss = train_model(dataloader_train, model, optimizer, loss_func)
#      print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}')

# accuracy, gold_labels, predictions = evaluate_model(model, dataloader_test)
     
# # ___________________________________________

# report = classification_report(gold_labels, predictions, target_names=target_names)
# print(report)

# show_confusion_matrix(gold_labels, predictions, target_names, model_name)