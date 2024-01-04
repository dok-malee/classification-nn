from sklearn.preprocessing import LabelEncoder
from letter_preprocessing import load_dataset, create_sparse_vectors, sparse_to_Tensor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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

def train_model(dataloader, model, optimizer, loss_func):
    running_loss = 0.0

    # Wrap train_loader with tqdm for the progress bar
    data = tqdm(dataloader, desc='Training', leave=False)

    for batch in data:
        inputs = batch[0]
        labels = batch[1]

        optimizer.zero_grad() 

        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(dataloader)


def evaluate_model(model, test_data):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_data:
            inputs = batch[0]
            labels = batch[1]

            outputs = model(inputs)
            pred = torch.argmax(outputs, 1)
            #x, preds = torch.max(outputs, 1)
            #print("argmax:", pred)
            #print("max:", x, preds)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    return accuracy
# _____________________________________________________________

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
# ['Wilhelm Busch', 'Henrik Ibsen', 'James Joyce', 'Franz Kafka', 'Friedrich Schiller', 'Johann Wolfgang von Goethe', 'Virginia Woolf']
label_enc = LabelEncoder()
labels_encoded_train = label_enc.fit_transform(train_author_labels)
labels_encoded_test = label_enc.transform(test_author_labels)

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

# training parameters
num_epochs = 20
learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
     train_loss = train_model(dataloader_train, model, optimizer, loss_func)
     print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}')

print(evaluate_model(model, dataloader_test))