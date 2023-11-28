import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


path_to_file = "/Users/sarahannauffelmann/desktop/01_WiSe23:24/Seminar Klassifizierung/Projekt FFNN/letters/classifier_data_train.jsonl"

list_of_letters = []
with open(path_to_file, "r") as file:
  for item in file:
    data = json.loads(item)
    list_of_letters.append(data)

num_letters = len(list_of_letters)

# Check what an instance looks like:
#print(list_of_letters[0])

def get_vocab(list_of_letters):
  vocab = set()
  for l in list_of_letters:
    text = l["text"]
    tokens = nltk.word_tokenize(text)
    for t in tokens:
      vocab.add(t)
  return vocab

# Check vocab for all letters:
letter_vocab = get_vocab(list_of_letters)
#print(get_vocab(list_of_letters))

# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py

# create_sparse_vectors
vectorizer = TfidfVectorizer(max_df=0.5, min_df=5) # should add stopwords
texts = [letter["text"] for letter in list_of_letters]
X_sparse = vectorizer.fit_transform(texts)

#print(X_sparse)

