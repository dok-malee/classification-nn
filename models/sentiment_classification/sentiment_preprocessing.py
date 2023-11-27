import collections
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# from sklearn.feature_extraction.text import CountVectorizer
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Embedding, Flatten, Dense, SpatialDropout1D, LSTM
# from keras.utils.np_utils import to_categorical



def nomalized_tokens(text):
    # tokenize the texts
    return word_tokenize(text.lower())


class DataInstance:
    def __init__(self, tokens, label,feature_counts):
        self.tokens = tokens
        self.label = label
        self.feature_counts = feature_counts

    @classmethod
    def from_dataset_instance(cls, instance):
        tokens = nomalized_tokens(instance['text'])
        label = instance['sentiment']
        feature_counts = collections.defaultdict(int)
        for feature in tokens:
            feature_counts[feature] += 1
        return cls(tokens, label, feature_counts)


class Dataset:
    def __init__(self, instance_list, vocab = None, n=None):
        self.instance_list = instance_list
        if vocab:
            self.feature_set = vocab
            self.set_feature_set(vocab=vocab)
        else:
            self.feature_set = set.union(*[set(inst.tokens) for inst in instance_list])
            if n:
                self.set_feature_set(n=n)

    @classmethod
    def from_file(cls, path, vocab=None, n=None):
        instance_list = []
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                data_inst = DataInstance.from_dataset_instance(json.loads(line))
                instance_list.append(data_inst)
        return cls(instance_list, vocab, n)

    def get_topn_features(self, n):
        """ This method returns a set with the top n frequent features. """

        featcounts = collections.Counter()
        for inst in self.instance_list:
            featcounts.update(inst.feature_counts)
        # remove the stopwords
        for stopw in stopwords.words('english'):
            if stopw in featcounts:
                del featcounts[stopw]

        topn_set = set([feat for feat, count in featcounts.most_common(n)])
        return topn_set

    def set_feature_set(self, vocab=None, n=None):
        """ limit the size of feature set to n and update the feature_counts of instances"""
        if vocab:
            self.feature_set = vocab
        elif n:
            self.feature_set = self.get_topn_features(n)
        for inst in self.instance_list:
            tmp_feature_counts = collections.Counter()
            tmp_tokens = []
            for feat in inst.feature_counts:
                if feat in self.feature_set:
                    tmp_feature_counts.update({feat: inst.feature_counts[feat]})
                    for i in range(inst.feature_counts[feat]):
                        tmp_tokens.append(feat)
            inst.feature_counts = tmp_feature_counts
            inst.tokens = tmp_tokens

    def generate_tfidf_matrix(self):
        texts = [' '.join(inst.tokens) for inst in self.instance_list]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        # print(vectorizer.vocabulary_)
        print(vectorizer.get_feature_names_out())

        return tfidf_matrix

    # def shuffle(self):
    #     random.shuffle(self.instance_list)

def build_vocab(dataset):
    vocab = set()
    vocab = dataset.feature_set
    return vocab




train_file = 'sentiment/classification/classification_sentiment_train.jsonl'
test_file = 'sentiment/classification/classification_sentiment_eval.jsonl'
n = 2000




train_data = Dataset.from_file(train_file, n=n)
vocabulary = build_vocab(train_data)
test_data = Dataset.from_file(test_file, vocab=vocabulary)

print(train_data.generate_tfidf_matrix())
print(test_data.generate_tfidf_matrix())


# class FeedForwardNeuralNetwork(nn.Module):
#     def __init__(self,self, input_size, hidden_size, num_classes):
#
#     def forward(self, x):


# Hyper Parameters
# input_size =
# hidden_size =
# num_classes = 2
# num_epochs =
# batch_size =
# learning_rate =