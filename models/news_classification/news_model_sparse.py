import wandb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from text_processing import create_sparse_matrices, load_datasets
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

train_file = '../../data/classification_data/data/news/classification/classification_news_train.jsonl'
test_file = '../../data/classification_data/data/news/classification/classification_news_eval.jsonl'
# print(torch.cuda.is_available())

wandb.init(project='news-classification-sparse-2')

word2vec_model_path = 'path/to/GoogleNews-vectors-negative300.bin'


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
X_train, X_test, y_train, y_test, feature_names, target_names = create_sparse_matrices(train_data=train_data,
                                                                                       test_data=test_data,
                                                                                       verbose=True)

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
hidden_size = 512
hidden_size2 = 512
output_size = len(target_names)
batch_size = 64
learning_rate = 0.001
epochs = 7

wandb.config.update({
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': output_size,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'epochs': epochs
})

# Create datasets and data loaders
train_dataset = SparseDataset(X_train_sparse, y_train)
test_dataset = SparseDataset(X_test_sparse, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = FFNN(input_size, hidden_size, hidden_size2, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# Log model architecture
wandb.watch(model)

# Training loop
for epoch in range(epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}')

    # Log training loss to WandB
    wandb.log({'train_loss': train_loss})

# Evaluate the model on the test set
all_preds, all_labels = evaluate_model(model, test_loader, device)

# Get classification report
classification_rep = classification_report(all_labels, all_preds, target_names=target_names, digits=4, output_dict=True)
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

plt.figure(figsize=(21, 18))
sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True, fmt=".4f", cmap="Blues")
plt.title("Classification Report")
plt.xlabel("Metrics")
plt.ylabel("Categories")

# Save the figure as a PNG
plt.savefig("news_sparse_report.png")


precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
wandb.log({'precision': precision, 'recall': recall, 'f1': f1})
wandb.finish()

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
plt.savefig('conf_matrix_sparse_bigram_e7.png')
