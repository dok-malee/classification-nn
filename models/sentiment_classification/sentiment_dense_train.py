import models.get_word_vectors
from models.get_word_vectors import load_datasets, get_tfidf_metrix, sparse_to_Tensor, get_word2vec_embeddings, get_transformer_embeddings
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
from nltk import word_tokenize
from tqdm import tqdm




def padding_texts(texts, word2id):
    padded_texts = []
    for text in texts:
        tokens = word_tokenize(text)
        seq = [word2id.get(token, word2id['<UNK>']) for token in tokens]

        zero_num = longest_seq - len(tokens)
        left_pad = zero_num // 2
        right_pad = zero_num - left_pad
        padded_texts.append([word2id['<PAD>']] * left_pad + seq + [word2id['<PAD>']] * right_pad)
    return torch.tensor(padded_texts)




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
sentiment_train_texts = sentiment_train_df['text'].tolist()[:1000]
sentiment_eval_texts = sentiment_eval_df['text'].tolist()

longest_seq = max(len(sent) for sent in sentiment_train_texts+sentiment_eval_texts)

# get word2vec metrics
word2vec_word2idx, word2vec_matrix = get_word2vec_embeddings(sentiment_train_texts)
train_word2vec_tensor = padding_texts(sentiment_train_texts, word2vec_word2idx)
eval_word2vec_tensor = padding_texts(sentiment_eval_texts, word2vec_word2idx)

"""get y"""
# encode the labels
y_train_sent = sentiment_train_df['sentiment'].tolist()[:1000]
y_eval_sent = sentiment_eval_df['sentiment'].tolist()

label_encoder = LabelEncoder()
train_encoded = label_encoder.fit_transform(y_train_sent)
eval_encoded = label_encoder.transform(y_eval_sent)

Y_train = torch.tensor(train_encoded)
Y_eval = torch.tensor(eval_encoded)
# print(Y_train[:5], Y_eval[:5])

"""tfidf_train"""
output_size = 2
# hyper-perameters for model
hidden_sizes = [128, 64]
# hyper-perameters for training
batch_size = 10
learning_rate = 0.01
num_epochs = 2000

word2vec_model = FFNN_with_embeddings(output_size=output_size, hidden_sizes=hidden_sizes, pretrained_emb=word2vec_matrix, vocabulary=word2vec_word2idx)


# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(word2vec_model.parameters(), lr=learning_rate)

train_model(train_word2vec_tensor, Y_train, word2vec_model, optimizer, criterion, batch_size, num_epochs)

"""check the accuracy on training and test data"""
Y_pred_train = predict(word2vec_model, train_word2vec_tensor, Y_train)
Y_pred_test = predict(word2vec_model, eval_word2vec_tensor, Y_eval)

accuracy_train = accuracy_score(Y_pred_train, Y_train)
accuracy_val = accuracy_score(Y_pred_test, Y_eval)

print("DenseEmbedding_Training accuracy", (accuracy_train)) #0.539
print("DenseEmbedding_Validation accuracy",(accuracy_val)) #0.529