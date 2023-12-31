from sklearn.preprocessing import LabelEncoder
from letter_preprocessing import load_dataset, create_sparse_vectors, sparse_to_Tensor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

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
# for X_batch, y_batch in dataloader_train:
#     print(X_batch, y_batch)

# ---------------------------------------------

# Define the Feedforward Neural Network model
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
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

def train_model(X, Y, model, opt, loss_func, batch_size, num_epochs):
    pass


# Code von Vertiefung
# train the model
# number_of_epochs = 2
# cbow_model = CBOW(emb_dim=embed_dim, vocab_dim=len(word2idx))
# optimizer = torch.optim.Adam(cbow_model.parameters(), lr=0.001)

# use dataloader to shuffle the data!!!
    # running_loss = 0.0
    # for epoch in range(number_of_epochs):
    #     #for i, data in tqdm(enumerate(input_data), total=len(list(input_data)), desc="Training"):
    #     for i, data in tqdm(enumerate(input_data)):
    #         instance_vector, label = data
    #         optimizer.zero_grad()
    #         output = cbow_model.forward(instance_vector)
    #         loss = cbow_model.loss(output, label)
    #         loss.backward() # stores the gradients of each parameter, can be called with grad()
    #         optimizer.step() # will use those stored gradients to optimize weights