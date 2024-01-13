from sklearn.preprocessing import LabelEncoder
from letter_preprocessing import load_dataset, get_word2vec_embeddings, get_word2vec_sent_embeddings
from author_classification_model_sparse import FFNN, train_model, show_confusion_matrix, evaluate_model, get_label_vectors
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

# Download GloVe embeddings from https://nlp.stanford.edu/projects/glove/
glove_file_path = "/Users/sarahannauffelmann/desktop/glove6B/glove.6B.300d.txt"

def load_glove_embeddings(file_path):
    """
    Returns:
    - embeddings_index (dict): A dictionary mapping words to their corresponding embedding vectors.
    """
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def create_glove_embeddings(texts, embeddings_index, vector_size=300):
    """
    Create GloVe embeddings for a list of texts using a pre-loaded embeddings dictionary.

    Parameters:
    - texts (list): A list of lists, where each inner list represents a tokenized sentence.
    - embeddings_index (dict): A dictionary mapping words to their corresponding embedding vectors.
    - vector_size (int): The dimensionality of the embedding vectors.

    Returns:
    - embeddings (torch.Tensor): A tensor containing the GloVe embeddings for each input text.
    """
    embeddings = []
    for sentence in texts:
        vectors = [embeddings_index.get(word, np.zeros(vector_size)) for word in sentence]
        if vectors:
            # Pad vectors with zeros to make them uniform in length
            max_len = max(len(vec) for vec in vectors)
            padded_vectors = [np.pad(vec, (0, max_len - len(vec))) for vec in vectors]
            embeddings.append(np.mean(padded_vectors, axis=0))
        else:
            embeddings.append(np.zeros(vector_size))
    return torch.tensor(embeddings, dtype=torch.float32)


if __name__ == "__main__":
    
    # paths to files
    path_to_train_file = "/Users/sarahannauffelmann/desktop/01_WiSe23:24/Seminar Klassifizierung/Projekt FFNN/letters/classifier_data_train.jsonl"
    path_to_test_file = "/Users/sarahannauffelmann/desktop/01_WiSe23:24/Seminar Klassifizierung/Projekt FFNN/letters/classifier_data_eval.jsonl"
        
    # training and test data
    train_inst, train_author_labels, train_lang_labels = load_dataset(path_to_train_file)
    test_inst, test_author_labels, test_lang_labels = load_dataset(path_to_test_file)

    # get glove embeddings
    emb = load_glove_embeddings(glove_file_path)
    X_train = create_glove_embeddings(train_inst, emb)
    X_test = create_glove_embeddings(test_inst, emb)

    # labels to vector encoding
    y_train, y_test, target_names = get_label_vectors(train_author_labels, test_author_labels)

    # create Dataloader
    # https://machinelearningmastery.com/training-a-pytorch-model-with-dataloader-and-dataset/
    dataloader_train = DataLoader(list(zip(X_train.to(device), y_train.to(device))), shuffle=True, batch_size=64)
    dataloader_test = DataLoader(list(zip(X_test.to(device), y_test.to(device))), shuffle=True, batch_size=64)
    #for X_batch, y_batch in dataloader_train:
        #print(X_batch, y_batch)

    # model parameters
    input_size = 300  # Dimensionality of dense embeddings glove.6B.300d
    output_size = 7         #  7 different authors
    hidden1 = 512
    hidden2 = 512

    model = FFNN(input_size, hidden1, hidden2, output_size).to(device)
    model_name = "FFNN_glove"

    # training parameters
    num_epochs = 25
    learning_rate = 0.001
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(num_epochs):
        train_loss = train_model(dataloader_train, model, optimizer, loss_func)
        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}')

    accuracy, gold_labels, predictions = evaluate_model(model, dataloader_test)
        
    report = classification_report(gold_labels, predictions, target_names=target_names)
    print(report)

    show_confusion_matrix(gold_labels, predictions, target_names, model_name)
