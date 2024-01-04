from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import sklearn
import collections
import json
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from typing import Dict
import nltk
from torch.nn.utils.rnn import pad_sequence
# nltk.download('punkt')

def load_datasets(train_file, test_file):
    """Load the news articles dataset."""

    # Load training data
    with open(train_file, 'r', encoding='utf-8') as file:
        train_data = [json.loads(line) for line in file]

    # Load test data
    with open(test_file, 'r', encoding='utf-8') as file:
        test_data = [json.loads(line) for line in file]

    return train_data, test_data

def get_vocab(train_texts, unk_threshold=0) -> Dict[str, int]:
    '''
    If directly use the embedding layer in the FFNN to get the embeddings, get_vocab function is called.
    Makes a dictionary of words given a list of tokenized sentences.
    :param train_texts: List of strings
    :param unk_threshold: All words below this count threshold are excluded from dictionary and replaced with UNK
    :return: A dictionary of string keys and index values
    '''
    tokenized_train_texts = [word_tokenize(sent) for sent in train_texts]
    # First count the frequency of each distinct ngram
    word_frequencies = {}
    for sent in tokenized_train_texts:
        for word in sent:
            if word not in word_frequencies:
                word_frequencies[word] = 0
            word_frequencies[word] += 1

    # Assign indices to each distinct ngram
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_frequencies.items():
        if freq > unk_threshold:  # only add words that are above threshold
            word_to_ix[word] = len(word_to_ix)
    # Print some info on dictionary size
    print(f"At unk_threshold={unk_threshold}, the dictionary contains {len(word_to_ix)} words")
    return word_to_ix

def get_tfidf_metrix(train_texts, eval_texts, hyper_param={'max_features': None, 'max_df': 1, 'min_df': 1}):
    """
    Get tf-idf sparse metrics of train_texts and test_texts using TfidfVectorizer

    Parameters:
        - train_texts: list of strings, text list of the training dataset.
        - eval_texts: list of strings, text list of the evaluation dataset.
    Returns:
        - output: pytorch tensors of word embeddings in each sentence
    """

    vectorizer = TfidfVectorizer(stop_words='english',
        max_features=hyper_param.get('max_features', None),
        max_df=hyper_param.get('max_df',1),
        min_df=hyper_param.get('min_df', 1))


    train_sparse_matrix = vectorizer.fit_transform(train_texts)
    eval_sparse_matrix = vectorizer.transform(eval_texts)
    vocab = enumerate(vectorizer.get_feature_names_out())

    return vocab, train_sparse_matrix, eval_sparse_matrix


def sparse_to_Tensor(sparse_features):
    """convert the sparse matrix and label lists into pytorch tensor"""
    sparse_tensor = torch.sparse_coo_tensor(
        torch.LongTensor(np.vstack(sparse_features.nonzero())),
        torch.FloatTensor(sparse_features.data),
        torch.Size(sparse_features.shape),
    )
    return sparse_tensor


def get_word2vec_embeddings(train_texts, vector_size=100, window=5, min_count=1, workers=4, vocab_size=10000):
    train_texts_sents = [word_tokenize(sent) for sent in train_texts]
    # print(train_texts_sents)
    '''if we use the word2vec matrix trained with training data on the test data:'''
    word2vec_model = Word2Vec(sentences=train_texts_sents, vector_size=vector_size, window=window, min_count=min_count,
             workers=workers, max_vocab_size=vocab_size)

    # Get embeddings for the vocabulary
    word2vec_word2idx = word2vec_model.wv.key_to_index # dictionary: word to index
    word2vec_word2idx['<UNK>'] = len(word2vec_word2idx)
    word2vec_word2idx['<PAD>'] = len(word2vec_word2idx)
    unk_token_embedding = np.zeros(word2vec_model.vector_size)
    pad_token_embedding = np.zeros(word2vec_model.vector_size)
    all_vectors = np.vstack([word2vec_model.wv.vectors, unk_token_embedding, pad_token_embedding])
    dense_embeddings = torch.tensor(all_vectors).to(torch.float32)
    # print(word2idx)
    return word2vec_word2idx, dense_embeddings



def get_transformer_embeddings(batch_texts):
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
    inputs = miniLM_tokenizer(batch_texts, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    # print("Tokenizer input shape:", inputs["input_ids"].shape)
    # print(inputs)
    outputs = miniLM_model(**inputs)

    return outputs

def get_tinybert_embeddings(texts):

    tinybert_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    tinybert_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    print("Tiny BERT Transformer loaded")
    # Use a dataloader to handle batching
    batch_size = 32
    dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)

    all_outputs = []
    for batch_texts in dataloader:
        inputs = tinybert_tokenizer(batch_texts, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        outputs = tinybert_model(**inputs)['last_hidden_state']
        all_outputs.append(outputs)

    # Concatenate the outputs for all batches
    embeddings = torch.cat(all_outputs, dim=0)

    return embeddings


if __name__ == "__main__":
    """usage examples"""

    # letter_train_path = '../data/classification_data/data/letters/classification/classifier_data_train.jsonl'
    # letter_eval_path = '../data/classification_data/data/letters/classification/classifier_data_eval.jsonl'
    #news_train_path = '../data/classification_data/data/news/classification/classification_news_train.jsonl'
    #news_eval_path = '../data/classification_data/data/news/classification/classification_news_eval.jsonl'
    sentiment_train_path = '../data/classification_data/data/sentiment/classification/classification_sentiment_train.jsonl'
    sentiment_eval_path = '../data/classification_data/data/sentiment/classification/classification_sentiment_eval.jsonl'

    """get lines of the three task datasets"""
    # letter_train_data, letter_eval_data = load_datasets(letter_train_path, letter_eval_path)
    #news_train_data, news_eval_data = load_datasets(news_train_path, news_eval_path)
    sentiment_train_data, sentiment_eval_data = load_datasets(sentiment_train_path, sentiment_eval_path)

    """get dataframes:"""
    # letter_train_df = pd.DataFrame(letter_train_data)
    # letter_eval_df = pd.DataFrame(letter_eval_data)
    #news_train_df = pd.DataFrame(news_train_data)
    # news_eval_df = pd.DataFrame(news_eval_data)
    sentiment_train_df = pd.DataFrame(sentiment_train_data)
    sentiment_eval_df = pd.DataFrame(sentiment_eval_data)

    """get the texts in train_dataset"""
    # letter_train_texts = letter_train_df['text'].tolist()
    # letter_eval_texts = letter_eval_df['text'].tolist()
    # news_train_headline = news_train_df['headline'].tolist()
    # news_eval_headline = news_eval_df['headline'].tolist()
    #news_train_description = news_train_df['short_description'].tolist()
    # news_eval_description = news_eval_df['short_description'].tolist()
    sentiment_train_texts = sentiment_train_df['text'].tolist()
    sentiment_eval_texts = sentiment_eval_df['text'].tolist()

    """get tfidf sparse tensors of train andn eval data"""
    # get_tfidf_metrix(letter_train_texts[:5], letter_eval_texts[:5])
    # get_tfidf_metrix(news_train_headline[:5], news_eval_headline[:5])
    # get_tfidf_metrix(news_train_description[:5], news_eval_description[:5])
    tfidf_vocab, sentiment_train_tfidf, sentiment_eval_tfidf = get_tfidf_metrix(sentiment_train_texts[:5], sentiment_eval_texts[:5])
    sentiment_train_tfidf_sparse = sparse_to_Tensor(sentiment_train_tfidf)
    print('sentiment_tf-df_sparse:', sentiment_train_tfidf_sparse)

    """get word2vec matrix from training data"""
    sentiment_word2vex_word2idx, sentiment_word2vec_matrix = get_word2vec_embeddings(sentiment_train_texts[:5])
    print("sentiment_word2vec:", sentiment_word2vex_word2idx, sentiment_word2vec_matrix)


    """get_transformer_embeddings of train and eval data"""
    # get_transformer_embeddings(letter_train_texts[:5])
    # get_transformer_embeddings(news_train_headline[:5])
    #gpu_properties = torch.cuda.get_device_properties(0)  # Use 0 if you have a single GPU
    #print(gpu_properties)
    # news_embeddings = get_tinybert_embeddings(news_train_description)
    # sentiment_train_LM_embeds = get_transformer_embeddings(sentiment_train_texts[:5])
    # print(sentiment_train_LM_embeds)





