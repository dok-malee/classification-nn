# Load and preprocess data
from transformers import DistilBertModel, DistilBertTokenizer, AutoTokenizer, AutoModel
import sklearn
import wandb
from tqdm import tqdm
import pandas as pd
from models.news_classification.text_processing import load_datasets, create_word2vec_embeddings
from models.get_word_vectors import get_transformer_embeddings
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

from models.sentiment_classification.new_sentiment_NNModel import FFNN_with_Transformer

# nltk.download('punkt')

train_file = '../../data/classification_data/data/news/classification/classification_news_train.jsonl'
test_file = '../../data/classification_data/data/news/classification/classification_news_eval.jsonl'

wandb.init(project='news-classification-sparse-2')


# Define FFNN_with_Transformer class
class FFNN_with_Transformer(nn.Module):
    def __init__(self, hidden_sizes, output_size, tokenizer_model, transformer_model):
        super(FFNN_with_Transformer, self).__init__()
        self.transformer_tokenizer = tokenizer_model
        self.transformer_model = transformer_model
        self.fc1 = nn.Linear(384, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.leakyrelu = nn.LeakyReLU()
        for param in self.transformer_model.encoder.layer.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = transformer_output[:, 0, :]
        x = self.leakyrelu(self.fc1(x))
        for layer in self.hidden_layers:
            x = self.leakyrelu(layer(x))
        x = self.output_layer(x)
        return x

# Function to get batches for transformer
def get_batches_for_transformer(texts, labels, batch_size, tokenizer, max_length=256, shuffle=True, device='cuda'):
    n = len(texts)
    indices = [i for i in range(n)]
    if shuffle:
        indices = sklearn.utils.shuffle(indices)

    for start_i in range(0, n, batch_size):
        end_i = min(n, start_i + batch_size)
        batch_indices = indices[start_i: end_i]
        batch_texts = [texts[i] for i in batch_indices]
        batch_labels = torch.tensor([labels[i] for i in batch_indices], dtype=torch.long).to(device)
        batch_inputs = tokenizer(batch_texts, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt").to(device)
        batch_input_ids = batch_inputs["input_ids"]
        batch_attention_mask = batch_inputs["attention_mask"]
        yield batch_input_ids, batch_attention_mask, batch_labels


# Function to predict with transformer
def predict_transformer(model, texts, labels, tokenizer, batch_size, shuffle=True, max_length=256):
    model.eval()
    batches = list(get_batches_for_transformer(texts, labels, batch_size=batch_size, tokenizer=tokenizer, shuffle=shuffle, max_length=max_length))
    predictions = []
    with torch.no_grad():
        for batch in batches:
            input_ids, mask_attention, labels = batch
            outputs = model(input_ids, mask_attention)
            _, prediction = torch.max(outputs, 1)
            predictions.extend(prediction.tolist())
    model.train()
    return predictions


if __name__ == '__main__':
    train_docs, test_docs = load_datasets(train_file, test_file)

    train_df = pd.DataFrame(train_docs)
    test_df = pd.DataFrame(test_docs)

    y_train = train_df['category']
    y_test = test_df['category']

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    train_texts = (train_df['headline'] + ' ' + train_df['short_description']).tolist()
    test_texts = (test_df['headline'] + ' ' + test_df['short_description']).tolist()

    input_size = 384  # Dimensionality of transformer output
    hidden_sizes = [512, 512]  # Hidden layer sizes
    output_size = len(np.unique(y_train))  # Number of unique classes
    batch_size = 32
    learning_rate = 0.001
    epochs = 25

    miniLM_tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
    miniLM_model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")

    ffnn_transformer = FFNN_with_Transformer(hidden_sizes=[512, 512], output_size=output_size, tokenizer_model=miniLM_tokenizer, transformer_model=miniLM_model)
    criterion_transformer = nn.CrossEntropyLoss()
    optimizer_transformer = optim.Adam(ffnn_transformer.parameters(), lr=learning_rate)
    scheduler_transformer = torch.optim.lr_scheduler.ExponentialLR(optimizer_transformer, gamma=0.9)

    wandb.init(project='news-classification-sparse-2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ffnn_transformer.to(device)
    wandb.watch(ffnn_transformer)

    for epoch in range(epochs):
        total_loss = 0.0
        train_input = list(get_batches_for_transformer(train_texts, y_train_encoded, batch_size=batch_size, tokenizer=miniLM_tokenizer, max_length=256, device=device))
        batch_num = len(train_input)
        for batch in train_input:
            optimizer_transformer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = ffnn_transformer(input_ids, attention_mask)
            loss = criterion_transformer(outputs, labels)
            loss.backward()
            optimizer_transformer.step()
            total_loss += loss.item()
        average_loss = total_loss / batch_num

        if (epoch + 1) % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, average_loss))
            wandb.log({'train_loss_transformer': average_loss})

    # Adjust the learning rate
    scheduler_transformer.step()

    # Predict with transformer on the test set
    all_preds_transformer_test = predict_transformer(ffnn_transformer, test_texts, y_test_encoded, miniLM_tokenizer, batch_size=batch_size, max_length=256)

    # Create classification report
    classification_rep = classification_report(y_test, all_preds_transformer_test, target_names=y_train.unique(), digits=4, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_test, all_preds_transformer_test, target_names=y_train.unique(), digits=4))

    plt.figure(figsize=(21, 18))
    sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True, fmt=".4f", cmap="Blues")
    plt.title("Classification Report")
    plt.xlabel("Metrics")
    plt.ylabel("Categories")

    # Save the figure as a PNG
    plt.savefig("news_transformer_report.png")

    precision_test = precision_score(y_test, all_preds_transformer_test, average='weighted')
    recall_test = recall_score(y_test, all_preds_transformer_test, average='weighted')
    f1_test = f1_score(y_test, all_preds_transformer_test, average='weighted')

    # Log evaluation metrics to WandB
    wandb.log({'precision': precision_test, 'recall': recall_test, 'f1': f1_test})
    wandb.finish()

    conf_matrix_dense = sk_confusion_matrix(y_test, all_preds_transformer_test, labels=range(output_size))
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
    plt.savefig('conf_matrix_transformer_512_512_e25.png')
