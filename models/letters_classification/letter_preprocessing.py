import json
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(path):
    "load the text and label from dataset"
    list_of_letters = []
    with open(path, "r", encoding="utf-8") as file:
        for item in file:
            data = json.loads(item)
            list_of_letters.append(data)
        
        instances = [letter["text"] for letter in list_of_letters]
        labels_author = [letter["author"] for letter in list_of_letters]    # for author classification
        labels_lang = [letter["lang"] for letter in list_of_letters]        # for language identification

    return instances, labels_author, labels_lang

def create_sparse_vectors(train_instances, test_instances, max_docfreq, min_docfreq):
    # https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
    "create tf-idf vector representations, instances: list of str"
    vectorizer = TfidfVectorizer(max_df=max_docfreq, min_df=min_docfreq) # no stopwords filter as we are dealing with a variety of languages 
    x_train = vectorizer.fit_transform(train_instances)
    x_test = vectorizer.transform(test_instances)

    feature_names = vectorizer.get_feature_names_out()
    #print(feature_names)
    return x_train, x_test

def sparse_to_Tensor(sparse_features):
    """convert the sparse matrix and label lists into pytorch tensor"""
    sparse_tensor = torch.sparse_coo_tensor(
        torch.LongTensor(np.vstack(sparse_features.nonzero())),
        torch.FloatTensor(sparse_features.data),
        torch.Size(sparse_features.shape),
    )
    return sparse_tensor

word2vec_size = 256
def get_word2vec_embeddings(train_texts, window=5, min_count=1, workers=4, vocab_size=10000):
    train_texts_sents = [word_tokenize(sent) for sent in train_texts]
    #print(train_texts_sents)
    '''if we use the word2vec matrix trained with training data on the test data:'''
    word2vec_model = Word2Vec(sentences=train_texts_sents, vector_size=word2vec_size, window=window, min_count=min_count,
             workers=workers, max_vocab_size=vocab_size)

    # Get embeddings for the vocabulary
    word2vec_word2idx = word2vec_model.wv.key_to_index # dictionary: word to index
    return word2vec_model, word2vec_word2idx

def get_word2vec_sent_embeddings(texts, word2vec_model, word2vec_word2idx):
    sent_embeddings = []
    for sentence in texts:
        word_vectors = [word2vec_model.wv[word] for word in sentence if word in word2vec_word2idx]
        if word_vectors:
            sent_embedding = np.mean(word_vectors, axis=0)
            sent_embeddings.append(sent_embedding)
        else:
            # If all words in the sentence are out-of-vocabulary, use a zero vector
            sent_embeddings.append(np.zeros(word2vec_size))

    dense_embeddings = torch.tensor(np.array(sent_embeddings)).to(torch.float32)
    return dense_embeddings





if __name__ == '__main__':
    # paths to files
    path_to_train_file = "/Users/sarahannauffelmann/desktop/01_WiSe23:24/Seminar Klassifizierung/Projekt FFNN/letters/classifier_data_train.jsonl"
    path_to_test_file = "/Users/sarahannauffelmann/desktop/01_WiSe23:24/Seminar Klassifizierung/Projekt FFNN/letters/classifier_data_eval.jsonl"
    
    # training and test data
    train_inst, train_author_labels, train_lang_labels = load_dataset(path_to_train_file)
    test_inst, test_author_labels, test_lang_labels = load_dataset(path_to_test_file)

    # sparse vectors
    x_train_sparse, x_test_sparse = create_sparse_vectors(train_inst, test_inst, 0.5, 5)

    # labels author classification
    y_train_author = train_author_labels
    y_test_author = test_author_labels

    # labels language identification
    y_train_lang= train_lang_labels
    y_test_lang = test_lang_labels

    word2vec_model, word2vec_word2idx = get_word2vec_embeddings(train_inst)
    #print(get_word2vec_sent_embeddings(train_inst, word2vec_model, word2vec_word2idx))