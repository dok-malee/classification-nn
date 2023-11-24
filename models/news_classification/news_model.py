import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import json


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    assert len(data) > 0, "No data loaded"
    return data


path = '../../data/classification_data/data/news/classification/classification_news_train.jsonl'
news_data = load_data(file_path=path)
