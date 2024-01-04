import models.get_word_vectors
from models.get_word_vectors import load_datasets, get_tfidf_metrix, sparse_to_Tensor, get_transformer_embeddings
from new_sentiment_NNModel import FFNN_with_tfidf, FFNN_with_Transformer, FFNN_with_embeddings, train_model, predict
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

sentiment_train_path = '../../data/classification_data/data/sentiment/classification/classification_sentiment_train.jsonl'
sentiment_eval_path = '../../data/classification_data/data/sentiment/classification/classification_sentiment_eval.jsonl'

# get lines of the three task datasets

sentiment_train_data, sentiment_eval_data = load_datasets(sentiment_train_path, sentiment_eval_path)

# get dataframes:

sentiment_train_df = pd.DataFrame(sentiment_train_data)
sentiment_eval_df = pd.DataFrame(sentiment_eval_data)

# get the texts in train_dataset and eval_dataset
# sentiment_train_texts = sentiment_train_df['text'].tolist()
# sentiment_eval_texts = sentiment_eval_df['text'].tolist()
sentiment_train_texts = sentiment_train_df['text'].tolist()[:5000]
sentiment_eval_texts = sentiment_eval_df['text'].tolist()

# get tfidf sparse metrics and transformer dense metrics
tfidf_vocab, train_tfidf, eval_tfidf = get_tfidf_metrix(sentiment_train_texts, sentiment_eval_texts, hyper_param={'max_features': 3000, 'max_df': 0.95, 'min_df': 0.005})
train_tfidf_tensor = sparse_to_Tensor(train_tfidf)
eval_tfidf_tensor = sparse_to_Tensor(eval_tfidf)


# train_lm_embeds = get_transformer_embeddings(sentiment_train_texts)
# eval_lm_embeds = get_transformer_embeddings(sentiment_eval_texts)
# print(train_lm_embeds[:5])

"""get y"""
# encode the labels
y_train_sent = sentiment_train_df['sentiment'].tolist()[:5000]
y_eval_sent = sentiment_eval_df['sentiment'].tolist()

label_encoder = LabelEncoder()
train_encoded = label_encoder.fit_transform(y_train_sent)
eval_encoded = label_encoder.transform(y_eval_sent)

Y_train = torch.tensor(train_encoded)
Y_eval = torch.tensor(eval_encoded)
# print(Y_train[:5], Y_eval[:5])

"""tfidf_train"""
input_size = train_tfidf_tensor.shape[1]
output_size = 2
# hyper-perameters for model
hidden_sizes = [128, 64]
# hyper-perameters for training
batch_size = 32
learning_rate = 0.05
num_epochs = 1000

tfidf_model = FFNN_with_tfidf(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes)

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tfidf_model.parameters(), lr=learning_rate)

train_model(train_tfidf_tensor, Y_train, tfidf_model, optimizer, criterion, batch_size, num_epochs)


"""check the accuracy on training and test data"""
Y_pred_train = predict(tfidf_model, train_tfidf_tensor)
Y_pred_test = predict(tfidf_model, eval_tfidf_tensor)

accuracy_train = accuracy_score(Y_pred_train, Y_train)
accuracy_val = accuracy_score(Y_pred_test, Y_eval)

print("Training accuracy", (accuracy_train)) #1.00
print("Validation accuracy",(accuracy_val)) #0.8248

