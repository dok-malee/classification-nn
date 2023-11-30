import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time


def create_sparse_vectors(data, text_column='headline', vectorizer=None):
    """
    Create sparse TF-IDF vectors for the text data.

    Parameters:
    - data: List of dictionaries where each dictionary represents a data point.
    - text_column: The name of the column containing the text data.
    - max_features: The maximum number of features (unique words) to consider in the TF-IDF vectorization.

    Returns:
    - X_sparse: Sparse TF-IDF matrix.
    """

    texts = [entry[text_column] for entry in data]

    if vectorizer is None:
        vectorizer = TfidfVectorizer()

    X_sparse = vectorizer.fit_transform(texts)

    return X_sparse, vectorizer


def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6


def load_dataset(train_file, test_file, verbose=False):
    """Load and vectorize the news articles dataset."""

    # Load training data
    with open(train_file, 'r', encoding='utf-8') as file:
        train_data = [json.loads(line) for line in file]

    # Load test data
    with open(test_file, 'r', encoding='utf-8') as file:
        test_data = [json.loads(line) for line in file]

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
