from sklearn.feature_extraction.text import TfidfVectorizer


def create_sparse_vectors(data, text_column='headline', max_features=5000):
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
    vectorizer = TfidfVectorizer(max_features=max_features)

    X_sparse = vectorizer.fit_transform(texts)

    return X_sparse
