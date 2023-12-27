# Load and preprocess data
import wandb
from tqdm import tqdm
from models.news_classification.text_processing import load_datasets, create_word2vec_embeddings
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, precision_score, recall_score, f1_score
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

wandb.init(project='news-classification-dense-w2v')


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

    # Create dense embeddings using word2vec
    dense_embeddings_train, dense_embeddings_val, dense_embeddings_test, y_train, y_val, y_test = create_word2vec_embeddings(train_data=train_docs,
                                                                                                test_data=test_docs,
                                                                                                vector_size=100)
    # Cast dense embeddings to torch.float32 to match the model's default data type
    dense_embeddings_train = dense_embeddings_train.to(torch.float32)
    dense_embeddings_test = dense_embeddings_test.to(torch.float32)
    dense_embeddings_val = dense_embeddings_val.to(torch.float32)

    # Print shape of dense embeddings
    print(f"Train Dense Embeddings Shape: {dense_embeddings_train.shape}")
    print(f"Test Dense Embeddings Shape: {dense_embeddings_test.shape}")

    input_size = 100  # Dimensionality of your dense embeddings
    hidden_size = 512
    hidden_size2 = 512
    output_size = len(np.unique(y_train))  # Use np.unique to get the number of unique classes
    batch_size = 64
    learning_rate = 0.001
    epochs = 25

    # Create datasets and data loaders for embeddings
    train_dataset_dense = DenseDataset(embeddings=dense_embeddings_train, labels=y_train)
    test_dataset_dense = DenseDataset(embeddings=dense_embeddings_test, labels=y_test)
    val_dataset_dense = DenseDataset(embeddings=dense_embeddings_val, labels=y_val)

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

        # Evaluate the model on val set
        all_preds_dense, all_labels_dense = evaluate_model(model_dense, val_loader, device, is_val=True)

        precision = precision_score(all_labels_dense, all_preds_dense, average='weighted')
        recall = recall_score(all_labels_dense, all_preds_dense, average='weighted')
        f1 = f1_score(all_labels_dense, all_preds_dense, average='weighted')

        # Log evaluation metrics to WandB
        wandb.log({'precision_val': precision, 'recall_val': recall, 'f1_val': f1})

        # Adjust the learning rate
        scheduler.step()


    # Evaluate the model on test set
    all_preds_dense_test, all_labels_dense_test = evaluate_model(model_dense, test_loader_dense, device)

    precision_test = precision_score(all_labels_dense_test, all_preds_dense_test, average='weighted')
    recall_test = recall_score(all_labels_dense_test, all_preds_dense_test, average='weighted')
    f1_test = f1_score(all_labels_dense_test, all_preds_dense_test, average='weighted')

    # Log evaluation metrics to WandB
    wandb.log({'precision_test': precision_test, 'recall_test': recall_test, 'f1_test': f1_test})
    wandb.finish()

    conf_matrix_dense = sk_confusion_matrix(all_labels_dense_test, all_preds_dense_test, labels=range(output_size))
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
    plt.savefig('conf_matrix_dense_headlines_512_512_e25.png')
