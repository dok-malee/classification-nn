from sklearn.preprocessing import LabelEncoder

from text_processing import create_sparse_matrices, load_datasets
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

train_file = '../../data/classification_data/data/news/classification/classification_news_train.jsonl'
test_file = '../../data/classification_data/data/news/classification/classification_news_eval.jsonl'
#print(torch.cuda.is_available())


# Custom dataset class
class SparseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'X': self.X[idx], 'y': torch.tensor(self.y[idx], dtype=torch.long)}  # Convert labels to PyTorch tensor


# Define the Feedforward Neural Network model
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    # Wrap train_loader with tqdm for the progress bar
    train_loader_tqdm = tqdm(train_loader, desc='Training', leave=False)

    for batch in train_loader_tqdm:
        inputs = batch['X'].to(device)
        labels = batch['y'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# Function to evaluate the model on the test set
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['X'].to(device)
            labels = batch['y'].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

train_data, test_data = load_datasets(train_file, test_file)
X_train, X_test, y_train, y_test, feature_names, target_names = create_sparse_matrices(train_data=train_data, test_data=test_data, verbose=True)

# Convert sparse matrices to PyTorch sparse tensors
X_train_sparse = torch.sparse_coo_tensor(
    torch.LongTensor(np.vstack(X_train.nonzero())),
    torch.FloatTensor(X_train.data),
    torch.Size(X_train.shape),
)

X_test_sparse = torch.sparse_coo_tensor(
    torch.LongTensor(np.vstack(X_test.nonzero())),
    torch.FloatTensor(X_test.data),
    torch.Size(X_test.shape),
)

# Hyperparameters
input_size = X_train_sparse.shape[1]
hidden_size = 256
output_size = len(target_names)
batch_size = 64
learning_rate = 0.001
epochs = 10

# Create datasets and data loaders
train_dataset = SparseDataset(X_train_sparse, y_train)
test_dataset = SparseDataset(X_test_sparse, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = FFNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# Training loop
for epoch in range(epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}')

# Evaluate the model on the test set
all_preds, all_labels = evaluate_model(model, test_loader, device)


# Compute the confusion matrix using sklearn
conf_matrix = sk_confusion_matrix(all_labels, all_preds, labels=range(output_size))

# Normalize the confusion matrix to get percentages
conf_matrix_percentage = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)

# Plot the confusion matrix
plt.figure(figsize=(21, 18))
sns.heatmap(conf_matrix_percentage, annot=True, fmt=".2%", cmap="Blues", xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('confusion_matrix_sparse_headlines.png')
