# Load and preprocess data
import wandb
from tqdm import tqdm
import pandas as pd
from models.news_classification.text_processing import load_datasets, create_word2vec_embeddings, \
    create_word2vec_embeddings2, load_datasets_bert
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModel, AutoTokenizer

# nltk.download('punkt')

# Load SBERT model and tokenizer
model_name = "bert-base-uncased"  # You can choose a different SBERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
sbert_model = AutoModel.from_pretrained(model_name)

train_file = '../../data/classification_data/data/news/classification/classification_news_train.jsonl'
test_file = '../../data/classification_data/data/news/classification/classification_news_eval.jsonl'

wandb.init(project='news-classification-sparse-2')

# Transfer learning with pretrained GoVe embeddings

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

    #if is_val:
    #    average_val_loss = running_val_loss / len(val_loader)
    #    wandb.log({'val_loss': average_val_loss})

    return all_preds, all_labels

def encode_text(docs):

    embeddings = []
    batch_size = 32  # Adjust the batch size based on your available memory
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        print(f"Encoding batch {i//batch_size + 1}/{len(docs)//batch_size + 1}")
        with torch.no_grad():
            outputs = sbert_model(**inputs)
        embeddings.append(outputs.pooler_output)
    return torch.cat(embeddings, dim=0)


if __name__ == '__main__':
    train_docs, test_docs, y_train, y_test = load_datasets_bert(train_file, test_file)

    # TODO: Retrieve input representation from a small pretrained transformer model like sbert to use as an input for my FNN
    #dense_embeddings_train = encode_text(train_docs)
    #dense_embeddings_test = encode_text(test_docs)

    # Cast dense embeddings to torch.float32 to match the model's default data type
    #dense_embeddings_train = dense_embeddings_train.to(torch.float32)
    #dense_embeddings_test = dense_embeddings_test.to(torch.float32)
    #dense_embeddings_val = dense_embeddings_val.to(torch.float32)

    # Save dense embeddings to file
    #torch.save(dense_embeddings_train, 'dense_embeddings_train.pt')
    #torch.save(dense_embeddings_test, 'dense_embeddings_test.pt')

    dense_embeddings_train = torch.load('dense_embeddings_train.pt')
    dense_embeddings_test = torch.load('dense_embeddings_test.pt')

    target_names = np.unique(y_train)

    # Print shape of dense embeddings
    print(f"Train Dense Embeddings Shape: {dense_embeddings_train.shape}")
    print(f"Test Dense Embeddings Shape: {dense_embeddings_test.shape}")

    input_size = 768
    hidden_size = 512
    hidden_size2 = 512
    output_size = len(np.unique(y_train))  # Use np.unique to get the number of unique classes
    batch_size = 64
    learning_rate = 0.001
    epochs = 25

    # Create datasets and data loaders for embeddings
    train_dataset_dense = DenseDataset(embeddings=dense_embeddings_train, labels=y_train)
    test_dataset_dense = DenseDataset(embeddings=dense_embeddings_test, labels=y_test)
    #val_dataset_dense = DenseDataset(embeddings=dense_embeddings_val, labels=y_val)

    train_loader = DataLoader(train_dataset_dense, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset_dense, batch_size=batch_size, shuffle=False)
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

        scheduler.step()


    # Evaluate the model on test set
    all_preds_dense_test, all_labels_dense_test = evaluate_model(model_dense, test_loader_dense, device)

    # Create classification report
    classification_rep = classification_report(all_labels_dense_test, all_preds_dense_test, target_names=target_names, digits=4, output_dict=True)
    print("Classification Report:")
    print(classification_report(all_labels_dense_test, all_preds_dense_test, target_names=target_names, digits=4))

    plt.figure(figsize=(21, 18))
    sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True, fmt=".4f", cmap="Blues")
    plt.title("Classification Report")
    plt.xlabel("Metrics")
    plt.ylabel("Categories")

    # Save the figure as a PNG
    plt.savefig("news_report_bert_e15.png")


    precision_test = precision_score(all_labels_dense_test, all_preds_dense_test, average='weighted')
    recall_test = recall_score(all_labels_dense_test, all_preds_dense_test, average='weighted')
    f1_test = f1_score(all_labels_dense_test, all_preds_dense_test, average='weighted')

    # Log evaluation metrics to WandB
    wandb.log({'precision': precision_test, 'recall': recall_test, 'f1': f1_test})
    wandb.finish()

    conf_matrix_dense = sk_confusion_matrix(all_labels_dense_test, all_preds_dense_test, labels=range(output_size))
    conf_matrix_percentage_dense = conf_matrix_dense / (conf_matrix_dense.sum(axis=1, keepdims=True) + 1e-10)

    # Plot the confusion matrix for dense embeddings
    plt.figure(figsize=(21, 18))
    sns.heatmap(conf_matrix_percentage_dense, annot=True, fmt=".2%", cmap="Blues", xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for Dense Embeddings')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig('conf_matrix_bert_e20.png')
