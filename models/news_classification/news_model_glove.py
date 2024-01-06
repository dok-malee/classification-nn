import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from models.news_classification.text_processing import load_datasets
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import torch.nn.functional as F

# Download GloVe embeddings from https://nlp.stanford.edu/projects/glove/
glove_file_path = 'glove_embeddings/glove.6B.300d.txt'

train_file = '../../data/classification_data/data/news/classification/classification_news_train.jsonl'
test_file = '../../data/classification_data/data/news/classification/classification_news_eval.jsonl'


wandb.init(project='news-classification-sparse-2')


class DenseDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'X': self.embeddings[idx], 'y': torch.tensor(self.labels[idx], dtype=torch.long)}


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


def load_glove_embeddings(file_path):
    """
    Load GloVe embeddings from a file.

    Parameters:
    - file_path (str): The path to the file containing GloVe embeddings.

    Returns:
    - embeddings_index (dict): A dictionary mapping words to their corresponding embedding vectors.
    """
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def create_glove_embeddings(texts, embeddings_index, vector_size=300):
    """
    Create GloVe embeddings for a list of texts using a pre-loaded embeddings dictionary.

    Parameters:
    - texts (list): A list of lists, where each inner list represents a tokenized sentence.
    - embeddings_index (dict): A dictionary mapping words to their corresponding embedding vectors.
    - vector_size (int): The dimensionality of the embedding vectors.

    Returns:
    - embeddings (torch.Tensor): A tensor containing the GloVe embeddings for each input text.
    """
    embeddings = []
    for sentence in texts:
        vectors = [embeddings_index.get(word, np.zeros(vector_size)) for word in sentence]
        if vectors:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(vector_size))
    return torch.tensor(embeddings, dtype=torch.float32)


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

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


def evaluate_model(model, test_loader, device, is_val=False):
    model.eval()
    all_preds = []
    all_labels = []
    running_val_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['X'].to(device)
            labels = batch['y'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if is_val:
        average_val_loss = running_val_loss / len(val_loader)
        wandb.log({'val_loss': average_val_loss})

    return all_preds, all_labels


if __name__ == '__main__':
    train_docs, test_docs = load_datasets(train_file, test_file)

    # Load GloVe embeddings
    glove_embeddings = load_glove_embeddings(glove_file_path)

    y_train = [doc['category'] for doc in train_docs]
    y_test = [doc['category'] for doc in test_docs]

    # Extract text  + ' ' + doc['short_description']
    train_texts = [word_tokenize(doc['headline'] + ' ' + doc['short_description']) for doc in train_docs]
    test_texts = [word_tokenize(doc['headline'] + ' ' + doc['short_description']) for doc in test_docs]

    # Split training data into training and validation sets
    train_texts, val_texts, y_train, y_val = train_test_split(train_texts, y_train, test_size=0.1, random_state=42)

    # Create GloVe embeddings for training, validation, and test data
    dense_embeddings_train = create_glove_embeddings(train_texts, glove_embeddings)
    dense_embeddings_val = create_glove_embeddings(val_texts, glove_embeddings)
    dense_embeddings_test = create_glove_embeddings(test_texts, glove_embeddings)

    input_size = 300  # Dimensionality of dense embeddings glove.6B.300d
    hidden_size = 512
    hidden_size2 = 512
    output_size = len(np.unique(y_train))  # Use np.unique to get the number of unique classes
    batch_size = 64
    learning_rate = 0.001
    epochs = 25

    # Create datasets and data loaders for embeddings
    train_dataset_dense = DenseDataset(embeddings=dense_embeddings_train, labels=y_train)
    val_dataset_dense = DenseDataset(embeddings=dense_embeddings_val, labels=y_val)
    test_dataset_dense = DenseDataset(embeddings=dense_embeddings_test, labels=y_test)

    train_loader = DataLoader(train_dataset_dense, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_dense, batch_size=batch_size, shuffle=False)
    test_loader_dense = DataLoader(test_dataset_dense, batch_size=batch_size, shuffle=False)

    model_dense = FFNN(input_size, hidden_size, hidden_size2, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_dense.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_dense.to(device)

    wandb.watch(model_dense)

    for epoch in range(epochs):
        train_loss = train_model(model_dense, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}')
        wandb.log({'train_loss': train_loss})

        # Evaluate the model on validation set
        all_preds_dense_val, all_labels_dense_val = evaluate_model(model_dense, val_loader, device, is_val=True)

        precision_val = precision_score(all_labels_dense_val, all_preds_dense_val, average='weighted')
        recall_val = recall_score(all_labels_dense_val, all_preds_dense_val, average='weighted')
        f1_val = f1_score(all_labels_dense_val, all_preds_dense_val, average='weighted')

        # Log evaluation metrics to WandB
        wandb.log({'precision_val': precision_val, 'recall_val': recall_val, 'f1_val': f1_val})

        # Adjust the learning rate
        scheduler.step()

    # Evaluate the model on test set after training is complete
    all_preds_dense_test, all_labels_dense_test = evaluate_model(model_dense, test_loader_dense, device)

    # Create classification report
    classification_rep = classification_report(all_labels_dense_test, all_preds_dense_test, digits=4, output_dict=True)
    print("Classification Report:")
    print(classification_report(all_labels_dense_test, all_preds_dense_test, digits=4))

    plt.figure(figsize=(21, 18))
    sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True, fmt=".4f", cmap="Blues")
    plt.title("Classification Report")
    plt.xlabel("Metrics")
    plt.ylabel("Categories")

    # Save the figure as a PNG
    plt.savefig("news_glove_report.png")

    precision_test = precision_score(all_labels_dense_test, all_preds_dense_test, average='weighted')
    recall_test = recall_score(all_labels_dense_test, all_preds_dense_test, average='weighted')
    f1_test = f1_score(all_labels_dense_test, all_preds_dense_test, average='weighted')

    # Log evaluation metrics to WandB
    wandb.log({'precision': precision_test, 'recall': recall_test, 'f1': f1_test})

    conf_matrix_dense = sk_confusion_matrix(all_labels_dense_test, all_preds_dense_test, labels=range(output_size))
    conf_matrix_percentage_dense = conf_matrix_dense / (conf_matrix_dense.sum(axis=1, keepdims=True) + 1e-10)

    # Plot the confusion matrix for dense embeddings
    plt.figure(figsize=(21, 18))
    sns.heatmap(conf_matrix_percentage_dense, annot=True, fmt=".2%", cmap="Blues", xticklabels=np.unique(y_train),
                yticklabels=np.unique(y_train))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for GloVe Embeddings')
    plt.tight_layout()
    plt.savefig('conf_matrix_glove.png')

    wandb.finish()
