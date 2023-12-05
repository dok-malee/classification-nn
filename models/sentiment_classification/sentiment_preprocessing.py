import collections
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np



def load_dataset(path):
    "load the text and label from dataset"
    with open(path, 'r', encoding="utf-8") as f:
        insts = []
        labels = []
        for line in f:
            insts.append(json.loads(line)['text'])
            labels.append(json.loads(line)['sentiment'])
        return insts, labels



def get_matrix_for_models(train_data, test_data, hyper_param={'max_features': None, 'max_df': 1, 'min_df': 1}):
    """
    hyper_param: dict of hyperparameters of TfidfVectorizer (hyper_param={'max_features': int, 'max_df': int, 'min_df': in})
    from file path generate train and test input and labels"""


    vectorizer = TfidfVectorizer(stop_words='english',
        max_features=hyper_param.get('max_features', None),
        max_df=hyper_param.get('max_df',1),
        min_df=hyper_param.get('min_df', 1))


    train_sparse_matrix = vectorizer.fit_transform(train_data[0])
    test_sparse_matrix = vectorizer.transform(test_data[0])
    features = vectorizer.get_feature_names_out()

    x_train = train_sparse_matrix
    y_train = train_data[1]

    x_test = test_sparse_matrix
    y_test = test_data[1]
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    train_path = '../../data/classification_data/data/sentiment/classification/classification_sentiment_train.jsonl'
    test_path = '../../data/classification_data/data/sentiment/classification/classification_sentiment_eval.jsonl'

    hyper_param = {'max_features': 100, 'max_df': 0.9}
    print(get_matrix_for_models(load_dataset(train_path), load_dataset(test_path), hyper_param)[0])





