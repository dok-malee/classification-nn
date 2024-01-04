import models.get_word_vectors
from models.get_word_vectors import load_datasets, get_tfidf_metrix, sparse_to_Tensor, get_word2vec_embeddings, get_transformer_embeddings
from new_sentiment_NNModel import FFNN_with_tfidf, FFNN_with_Transformer, FFNN_with_embeddings, train_model, train_model_transformer, predict, predict_transformer
import sklearn
import collections
import json
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from nltk import word_tokenize
from tqdm import tqdm


sentiment_train_path = '../../data/classification_data/data/sentiment/classification/classification_sentiment_train.jsonl'
sentiment_eval_path = '../../data/classification_data/data/sentiment/classification/classification_sentiment_eval.jsonl'

# get lines of the three task datasets

sentiment_train_data, sentiment_eval_data = load_datasets(sentiment_train_path, sentiment_eval_path)

# get dataframes:

sentiment_train_df = pd.DataFrame(sentiment_train_data)
sentiment_eval_df = pd.DataFrame(sentiment_eval_data)

"""get the texts in train_dataset and eval_dataset"""
# sentiment_train_texts = sentiment_train_df['text'].tolist()
# sentiment_eval_texts = sentiment_eval_df['text'].tolist()
sentiment_train_texts = sentiment_train_df['text'].tolist()[:1000]
sentiment_eval_texts = sentiment_eval_df['text'].tolist()[:100]


"""get y"""
# encode the labels
y_train_sent = sentiment_train_df['sentiment'].tolist()[:1000]
y_eval_sent = sentiment_eval_df['sentiment'].tolist()[:100]

label_encoder = LabelEncoder()
train_encoded = label_encoder.fit_transform(y_train_sent)
eval_encoded = label_encoder.transform(y_eval_sent)

Y_train = torch.tensor(train_encoded)
Y_eval = torch.tensor(eval_encoded)


"""transformer_train"""
output_size = 2
# hyper-perameters for model
hidden_sizes = [128]
# hyper-perameters for training
batch_size = 10
learning_rate = 0.01
num_epochs = 20

transformer_model = FFNN_with_Transformer(output_size=output_size, hidden_sizes=hidden_sizes)

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(transformer_model.parameters(), lr=learning_rate)

train_model_transformer(sentiment_train_texts, Y_train, transformer_model, optimizer, criterion, batch_size, num_epochs)


"""check the accuracy on training and test data"""
Y_pred_train = predict_transformer(transformer_model, sentiment_train_texts, Y_train, batch_size=32)
Y_pred_test = predict_transformer(transformer_model, sentiment_eval_texts, Y_eval, batch_size=32)

accuracy_train = accuracy_score(Y_pred_train, Y_train)
accuracy_val = accuracy_score(Y_pred_test, Y_eval)

print("Transformer_Training accuracy", (accuracy_train)) #1.00
print("Transformer_Validation accuracy",(accuracy_val)) #0.813

