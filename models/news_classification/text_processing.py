import json
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from transformers import BertTokenizer, BertModel
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np


def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6


def load_datasets(train_file, test_file):
    """Load the news articles dataset."""

    # Load training data
    with open(train_file, 'r', encoding='utf-8') as file:
        train_data = [json.loads(line) for line in file]

    # Load test data
    with open(test_file, 'r', encoding='utf-8') as file:
        test_data = [json.loads(line) for line in file]

    return train_data, test_data


def create_sparse_matrices(train_data, test_data, verbose=False):
    """Vectorize the news articles dataset."""

    # Create DataFrames for easier handling
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Extract labels
    y_train = train_df['category']
    y_test = test_df['category']

    # Use only the headline for each article
    train_docs = train_df['headline']
    test_docs = test_df['headline']

    # Extracting features from the training data using a sparse vectorizer
    t0 = time()
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english")
    X_train = vectorizer.fit_transform(train_docs)
    duration_train = time() - t0

    # Extracting features from the test data using the same vectorizer
    t0 = time()
    X_test = vectorizer.transform(test_docs)
    duration_test = time() - t0

    feature_names = vectorizer.get_feature_names_out()

    if verbose:
        # Compute size of loaded data
        data_train_size_mb = size_mb(train_docs)
        data_test_size_mb = size_mb(test_docs)

        print(f"{len(train_docs)} documents - {data_train_size_mb:.2f}MB (training set)")
        print(f"{len(test_docs)} documents - {data_test_size_mb:.2f}MB (test set)")
        print(f"{len(y_train.unique())} categories")
        print(f"Vectorize training done in {duration_train:.3f}s at {data_train_size_mb / duration_train:.3f}MB/s")
        print(f"n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
        print(f"Vectorize testing done in {duration_test:.3f}s at {data_test_size_mb / duration_test:.3f}MB/s")
        print(f"n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")

    return X_train, X_test, y_train, y_test, feature_names, y_train.unique()


def create_dense_embeddings(train_data, test_data, text_column='headline', model_name='bert-base-uncased',
                            device='cuda'):
    """
    Create dense embeddings using a pre-trained transformer model (e.g., BERT).

    Parameters:
    - data: List of dictionaries where each dictionary represents a data point.
    - text_column: The name of the column containing the text data.
    - model_name: The name of the pre-trained transformer model.

    Returns:
    - embeddings: Dense embeddings matrix.
    """
    # Create DataFrames for easier handling
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Extract labels
    y_train = train_df['category']
    y_test = test_df['category']

    # Extract text
    train_texts = train_df[text_column]
    test_texts = test_df[text_column]

    # Load pre-trained tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)

    # Tokenize and obtain dense embeddings for training data
    tokenized_train_texts = tokenizer(train_texts.tolist(), padding=True, truncation=True, return_tensors="pt").to(
        device)
    with torch.no_grad():
        outputs_train = model(**tokenized_train_texts)
    dense_embeddings_train = outputs_train.last_hidden_state

    # Tokenize and obtain dense embeddings for test data
    tokenized_test_texts = tokenizer(test_texts.tolist(), padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_test = model(**tokenized_test_texts)
    dense_embeddings_test = outputs_test.last_hidden_state

    return dense_embeddings_train, dense_embeddings_test, y_train, y_test


def create_word2vec_embeddings(train_data, test_data, text_column='headline', vector_size=100, window=5, min_count=1,
                               workers=4):
    """
    Create Word2Vec embeddings for news classification.

    Parameters:
    - train_data: List of dictionaries representing training data.
    - test_data: List of dictionaries representing test data.
    - text_column: The name of the column containing the text data.
    - vector_size: Dimensionality of the word embeddings.
    - window: Maximum distance between the current and predicted word within a sentence.
    - min_count: Ignores all words with a total frequency lower than this.
    - workers: Number of CPU cores to use for training.

    Returns:
    - dense_embeddings_train: Word2Vec embeddings for training data.
    - dense_embeddings_test: Word2Vec embeddings for test data.
    - y_train: Labels for training data.
    - y_test: Labels for test data.
    """

    # Create DataFrames for easier handling
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Extract labels
    y_train = train_df['category']
    y_test = test_df['category']

    # Extract text
    train_texts = [word_tokenize(text) for text in train_df[text_column]]
    test_texts = [word_tokenize(text) for text in test_df[text_column]]

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=train_texts, vector_size=vector_size, window=window, min_count=min_count,
                              workers=workers)

    # Get embeddings for training data
    dense_embeddings_train = []
    for sentence in train_texts:
        embeddings = [word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv]
        if embeddings:
            dense_embeddings_train.append(np.mean(embeddings, axis=0))
        else:
            dense_embeddings_train.append(np.zeros(vector_size))

    dense_embeddings_train = torch.tensor(dense_embeddings_train)

    # Get embeddings for test data
    dense_embeddings_test = []
    for sentence in test_texts:
        embeddings = [word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv]
        if embeddings:
            dense_embeddings_test.append(np.mean(embeddings, axis=0))
        else:
            # might encounter words in the test data (or any unseen data) that were not part of the training vocabulary
            dense_embeddings_test.append(np.zeros(vector_size))

    dense_embeddings_test = torch.tensor(dense_embeddings_test)

    return dense_embeddings_train, dense_embeddings_test, y_train, y_test
