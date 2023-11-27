import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import json
from text_processing import create_sparse_vectors


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    assert len(data) > 0, "No data loaded"
    return data


def create_labels(loaded_data):
    labels = [entry['category'] for entry in loaded_data]
    label_encoder = LabelEncoder()

    # fits the encoder to the unique labels in your data and transforms the labels into numerical values
    # each color is assigned an integer
    y = label_encoder.fit_transform(labels)
    y_onehot = torch.nn.functional.one_hot(torch.tensor(y))

    label_mapping = dict(zip(label_encoder.classes_, y_onehot.numpy()))
    return y_onehot, label_mapping


if __name__ == "__main__":
    path = '../../data/classification_data/data/news/classification/classification_news_train.jsonl'
    news_data = load_data(file_path=path)

    #X_sparse = create_sparse_vectors(news_data)
    #print(X_sparse)

    labels_onehot, labels_mapping = create_labels(news_data)
    print(labels_mapping)
