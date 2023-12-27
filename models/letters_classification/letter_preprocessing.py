import json
from sklearn.feature_extraction.text import TfidfVectorizer


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


def create_sparse_vectors(instances, max_docfreq, min_docfreq):
    # https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
    "create tf-idf vector representations, instances: list of str"
    vectorizer = TfidfVectorizer(max_df=max_docfreq, min_df=min_docfreq) # no stopwords filter as we are dealing with a variety of languages 
    X_sparse = vectorizer.fit_transform(instances)
    return X_sparse


if __name__ == '__main__':
    # paths to files
    path_to_train_file = "/Users/sarahannauffelmann/desktop/01_WiSe23:24/Seminar Klassifizierung/Projekt FFNN/letters/classifier_data_train.jsonl"
    path_to_test_file = "/Users/sarahannauffelmann/desktop/01_WiSe23:24/Seminar Klassifizierung/Projekt FFNN/letters/classifier_data_eval.jsonl"
    
    # training data
    train_inst, train_author_labels, train_lang_labels = load_dataset(path_to_train_file)
    
    # test data
    test_inst, test_author_labels, test_lang_labels = load_dataset(path_to_test_file)

    # sparse vectors
    train_vectors_sparse = create_sparse_vectors(train_inst, 0.5, 5)
    test_vectors_sparse = create_sparse_vectors(test_inst, 0.5, 5)
    #print(train_vectors_sparse)
    #print(test_vectors_sparse)