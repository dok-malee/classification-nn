import preprocessing
from preprocessing import load_datasets, get_tfidf_matrix, sparse_to_Tensor
from sentiment_NNModel import FFNN, train_model, predict, get_classification_report, show_confusion_matrix
import sklearn
import collections
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import wandb


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
sentiment_train_texts = sentiment_train_df['text'].tolist()
sentiment_eval_texts = sentiment_eval_df['text'].tolist()

# get tfidf sparse metrics and transformer dense metrics
tfidf_vocab, train_tfidf, eval_tfidf = get_tfidf_matrix(sentiment_train_texts, sentiment_eval_texts, ngram=(2,2), hyper_param={'max_features': 3000, 'max_df': 0.99, 'min_df': 0.005})
train_tfidf_tensor = sparse_to_Tensor(train_tfidf)
eval_tfidf_tensor = sparse_to_Tensor(eval_tfidf)
"""split the train and val set"""

"""get y"""
# encode the labels
label_train_sent = sentiment_train_df['sentiment'].tolist()
label_eval_sent = sentiment_eval_df['sentiment'].tolist()

label_encoder = LabelEncoder()
train_encoded = label_encoder.fit_transform(label_train_sent)
eval_encoded = label_encoder.transform(label_eval_sent)

Y_train = torch.tensor(train_encoded)
Y_eval = torch.tensor(eval_encoded)
# print(Y_train[:5], Y_eval[:5])
label_class_names = label_encoder.classes_


"""tfidf_train"""
input_size = train_tfidf_tensor.shape[1]
output_size = 2
# hyper-perameters for model
hidden_sizes = [128, 64]
# hyper-perameters for training
batch_size = 32
learning_rate = 0.05
num_epochs = 20

tfidf_model = FFNN(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes)

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tfidf_model.parameters(), lr=learning_rate)

wandb.init(project="sentiment_classification_with_bigram_tfidf_vectors")
train_model(train_tfidf_tensor, Y_train, tfidf_model, optimizer, criterion, batch_size, num_epochs, print_epoch=2)
wandb.finish()
best_model_with_tfidf = tfidf_model

"""check the accuracy on training and test data"""
Y_pred_train = predict(best_model_with_tfidf, train_tfidf_tensor)
Y_pred_eval = predict(best_model_with_tfidf, eval_tfidf_tensor)


train_report = get_classification_report(Y_train, Y_pred_train, label_class_names)
eval_report = get_classification_report(Y_eval, Y_pred_eval, label_class_names)
print(train_report)
print(eval_report)

show_confusion_matrix(Y_train, Y_pred_train, label_class_names, "Tfidf_Bigram_Train")
show_confusion_matrix(Y_eval, Y_pred_eval, label_class_names, "Tfidf_Bigram_Test")