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


def create_sparse_vectors(train_instances, test_instances, max_docfreq, min_docfreq):
    # https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
    "create tf-idf vector representations, instances: list of str"
    vectorizer = TfidfVectorizer(max_df=max_docfreq, min_df=min_docfreq) # no stopwords filter as we are dealing with a variety of languages 
    x_train = vectorizer.fit_transform(train_instances)
    x_test = vectorizer.transform(test_instances)

    feature_names = vectorizer.get_feature_names_out()
    #print(feature_names)
    return x_train, x_test

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

    