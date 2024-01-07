import preprocessing
from preprocessing import load_datasets, get_glove_sentence_embedding, glove_model
from sentiment_NNModel import FFNN, train_model, predict, get_classification_report, show_confusion_matrix
import sklearn
import collections
import json
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from nltk import word_tokenize
import wandb





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
sentiment_train_texts = sentiment_train_df['text'].tolist()
sentiment_eval_texts = sentiment_eval_df['text'].tolist()

# get word2vec metrics
train_glove_tensor = get_glove_sentence_embedding(sentiment_train_texts, glove=glove_model)
eval_glove_tensor = get_glove_sentence_embedding(sentiment_eval_texts, glove=glove_model)

"""get y"""
# encode the labels
y_train_sent = sentiment_train_df['sentiment'].tolist()
y_eval_sent = sentiment_eval_df['sentiment'].tolist()

label_encoder = LabelEncoder()
train_encoded = label_encoder.fit_transform(y_train_sent)
eval_encoded = label_encoder.transform(y_eval_sent)

Y_train = torch.tensor(train_encoded)
Y_eval = torch.tensor(eval_encoded)
# print(Y_train[:5], Y_eval[:5])
label_class_names = label_encoder.classes_

# model = FFNN_with_tfidf(output_size=output_size, hidden_sizes=hidden_sizes, pretrained_emb=word2vec_matrix, vocabulary=word2vec_word2idx)
"""word2vec_train"""
input_size = glove_model.dim
output_size = 2
# hyper-perameters for model
hidden_sizes = [128, 64]
# hyper-perameters for training
batch_size = 64
learning_rate = 0.05
num_epochs = 500

model = FFNN(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes)

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

wandb.init(project="sentiment_classification_with_GloVe_embeddings")
train_model(train_glove_tensor, Y_train, model, optimizer, criterion, batch_size, num_epochs)
wandb.finish()


best_model_with_glove = model
"""check the accuracy on training and test data"""
Y_pred_train = predict(best_model_with_glove, train_glove_tensor)
Y_pred_eval = predict(best_model_with_glove, eval_glove_tensor)



train_report = get_classification_report(Y_train, Y_pred_train, label_class_names)
eval_report = get_classification_report(Y_eval, Y_pred_eval, label_class_names)
print(train_report)
print(eval_report)

show_confusion_matrix(Y_train, Y_pred_train, label_class_names, "GloVe_Train")
show_confusion_matrix(Y_eval, Y_pred_eval, label_class_names, "GloVe_Test")

