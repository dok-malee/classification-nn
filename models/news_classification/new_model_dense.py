# Load and preprocess data
from tqdm import tqdm
from models.news_classification.text_processing import load_datasets, create_word2vec_embeddings
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

# nltk.download('punkt')

train_file = '../../data/classification_data/data/news/classification/classification_news_train.jsonl'
test_file = '../../data/classification_data/data/news/classification/classification_news_eval.jsonl'


class DenseDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'X': self.embeddings[idx], 'y': torch.tensor(self.labels[idx], dtype=torch.long)}


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


if __name__ == '__main__':
    train_docs, test_docs = load_datasets(train_file, test_file)

    # Create dense embeddings using word2vec
    dense_embeddings_train, dense_embeddings_test, y_train, y_test = create_word2vec_embeddings(train_data=train_docs,
                                                                                                test_data=test_docs,
                                                                                                text_column='headline',
                                                                                                vector_size=100)

    # Cast dense embeddings to torch.float32 to match the model's default data type
    dense_embeddings_train = dense_embeddings_train.to(torch.float32)
    dense_embeddings_test = dense_embeddings_test.to(torch.float32)

    # Print shape of dense embeddings
    print(f"Train Dense Embeddings Shape: {dense_embeddings_train.shape}")
    print(f"Test Dense Embeddings Shape: {dense_embeddings_test.shape}")

    input_size = 100  # Dimensionality of your dense embeddings
    hidden_size = 256
    output_size = len(np.unique(y_train))  # Use np.unique to get the number of unique classes
    batch_size = 64
    learning_rate = 0.001
    epochs = 10

    # Create datasets and data loaders for embeddings
    train_dataset_dense = DenseDataset(embeddings=dense_embeddings_train, labels=y_train)
    test_dataset_dense = DenseDataset(embeddings=dense_embeddings_test, labels=y_test)

    train_loader = DataLoader(train_dataset_dense, batch_size=batch_size, shuffle=True)
    test_loader_dense = DataLoader(test_dataset_dense, batch_size=batch_size, shuffle=False)

    model_dense = FFNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_dense.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_dense.to(device)

    for epoch in range(epochs):
        train_loss = train_model(model_dense, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}')

    # Evaluate the model on test set
    all_preds_dense, all_labels_dense = evaluate_model(model_dense, test_loader_dense, device)

    conf_matrix_dense = sk_confusion_matrix(all_labels_dense, all_preds_dense, labels=range(output_size))
    conf_matrix_percentage_dense = conf_matrix_dense / (conf_matrix_dense.sum(axis=1, keepdims=True) + 1e-10)

    # Plot the confusion matrix for dense embeddings
    plt.figure(figsize=(21, 18))
    sns.heatmap(conf_matrix_percentage_dense, annot=True, fmt=".2%", cmap="Blues", xticklabels=y_train.unique(),
                yticklabels=y_train.unique())
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for Dense Embeddings')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig('confusion_matrix_dense_embeddings_headlines.png')
