import json
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from gensim.models import Word2Vec, KeyedVectors
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


def create_sparse_matrices(train_data, test_data, verbose=False, text_columns=['headline', 'short_description']):
    """Vectorize the news articles dataset."""

    # Create DataFrames for easier handling
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Extract labels
    y_train = train_df['category']
    y_test = test_df['category']

    # Use only one source for each article
    # train_docs = train_df[text_columns[0]]
    # test_docs = test_df[text_columns[0]]

    # Merge text from specified columns
    train_docs = train_df[text_columns[0]] + ' ' + train_df[text_columns[1]]
    test_docs = test_df[text_columns[0]] + ' ' + test_df[text_columns[1]]

    # Extracting features from the training data using a sparse vectorizer
    t0 = time()
    vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=1.0, min_df=1, stop_words="english", ngram_range=(2, 2))
    # per default each row will be already normalized
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


def load_pretrained_word2vec_model(path):
    return KeyedVectors.load_word2vec_format(path, binary=True)


def create_word2vec_embeddings(train_data, test_data, text_column=['headline', 'short_description'], vector_size=300,
                               window=10, min_count=1,
                               workers=4, validation_size=0.1):
    """
    Create Word2Vec embeddings for news classification.
    Uses skip-gram model per default.

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
    - dense_embeddings_val: Word2Vec embeddings for validation data.
    - dense_embeddings_test: Word2Vec embeddings for test data.
    - y_train: Labels for training data.
    - y_val: Labels for validation data.
    - y_test: Labels for test data.
    - y_train.unique(): Set of Labels for training data
    """

    # Create DataFrames for easier handling
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Extract labels
    y_train = train_df['category']
    y_test = test_df['category']

    # Extract text: Use this if you only have one source
    # train_texts = [word_tokenize(row[text_column]) for _, row in train_df.iterrows()]
    # test_texts = [word_tokenize(row[text_column]) for _, row in test_df.iterrows()]

    # multiple sources merged
    train_texts = [word_tokenize(row[text_column[0]] + ' ' + row[text_column[1]]) for _, row in train_df.iterrows()]
    test_texts = [word_tokenize(row[text_column[0]] + ' ' + row[text_column[1]]) for _, row in test_df.iterrows()]

    train_texts, val_texts, y_train, y_val = train_test_split(train_texts, y_train, test_size=validation_size,
                                                              random_state=42)

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

    # Get embeddings for validation data
    dense_embeddings_val = []
    for sentence in val_texts:
        embeddings = [word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv]
        if embeddings:
            dense_embeddings_val.append(np.mean(embeddings, axis=0))
        else:
            dense_embeddings_val.append(np.zeros(vector_size))

    dense_embeddings_val = torch.tensor(dense_embeddings_val)

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

    return dense_embeddings_train, dense_embeddings_val, dense_embeddings_test, y_train, y_val, y_test, y_train.unique()


word2vec_model_path = 'word2vec_embeddings/GoogleNews-vectors-negative300.bin'
pretrained_word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)


def create_word2vec_embeddings2(train_data, test_data, text_column=['headline', 'short_description'], vector_size=300,
                                validation_size=0.1):
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    y_train = train_df['category']
    y_test = test_df['category']

    train_texts = [word_tokenize(row[text_column[0]] + ' ' + row[text_column[1]]) for _, row in train_df.iterrows()]
    test_texts = [word_tokenize(row[text_column[0]] + ' ' + row[text_column[1]]) for _, row in test_df.iterrows()]

    train_texts, val_texts, y_train, y_val = train_test_split(train_texts, y_train, test_size=validation_size,
                                                              random_state=42)

    # Get embeddings for training data
    dense_embeddings_train = get_embeddings(train_texts, pretrained_word2vec_model, vector_size)

    # Get embeddings for validation data
    dense_embeddings_val = get_embeddings(val_texts, pretrained_word2vec_model, vector_size)

    # Get embeddings for test data
    dense_embeddings_test = get_embeddings(test_texts, pretrained_word2vec_model, vector_size)

    return dense_embeddings_train, dense_embeddings_val, dense_embeddings_test, y_train, y_val, y_test, y_train.unique()


def get_embeddings(texts, word2vec_model, vector_size):
    dense_embeddings = []
    for sentence in texts:
        embeddings = [word2vec_model[word] for word in sentence if word in word2vec_model]
        if embeddings:
            dense_embeddings.append(np.mean(embeddings, axis=0))
        else:
            dense_embeddings.append(np.zeros(vector_size))
    return torch.tensor(dense_embeddings)


def get_transformer_embeddings(texts):
    """
    Get word embeddings using Pre-Trained Transformer miniLM

    Parameters:
        - texts: list of strings, text list of the training or evaluation dataset.
    Returns:
        - output: pytorch tensors of word embeddings in each sentence
    """

    miniLM_tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
    miniLM_model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
    print("miniLM Transformer loaded")
    inputs = miniLM_tokenizer(texts, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    # print("Tokenizer input shape:", inputs["input_ids"].shape)
    # print(inputs)
    outputs = miniLM_model(**inputs)

    return outputs
