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


if __name__ == "__main__":
    # paths to files
    path_to_train_file = "/Users/sarahannauffelmann/desktop/01_WiSe23:24/Seminar Klassifizierung/Projekt FFNN/letters/classifier_data_train.jsonl"
    path_to_test_file = "/Users/sarahannauffelmann/desktop/01_WiSe23:24/Seminar Klassifizierung/Projekt FFNN/letters/classifier_data_eval.jsonl"
        
    # training and test data
    train_inst, train_author_labels, train_lang_labels = load_dataset(path_to_train_file)
    test_inst, test_author_labels, test_lang_labels = load_dataset(path_to_test_file)

    # create embeddings
    word2vec_size = 256
    word2vec_model, word2vec_word2id = get_word2vec_embeddings(train_inst)
    X_train = get_word2vec_sent_embeddings(train_inst, word2vec_model, word2vec_word2id)
    X_test = get_word2vec_sent_embeddings(test_inst, word2vec_model, word2vec_word2id)  

    # labels to vector encoding
    y_train, y_test, target_names = get_label_vectors(train_author_labels, test_author_labels)

    # create Dataloader
    # https://machinelearningmastery.com/training-a-pytorch-model-with-dataloader-and-dataset/
    dataloader_train = DataLoader(list(zip(X_train.to(device), y_train.to(device))), shuffle=True, batch_size=64)
    dataloader_test = DataLoader(list(zip(X_test.to(device), y_test.to(device))), shuffle=True, batch_size=64)
    #for X_batch, y_batch in dataloader_train:
        #print(X_batch, y_batch)

    # model parameters
    input_size = word2vec_size
    output_size = 7         #  7 different authors
    hidden1 = 512
    hidden2 = 512

    model = FFNN(input_size, hidden1, hidden2, output_size).to(device)
    model_name = "FFNN_sparse"

    # training parameters
    num_epochs = 30
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
