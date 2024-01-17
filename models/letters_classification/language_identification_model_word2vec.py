from letter_preprocessing import load_dataset, get_word2vec_embeddings, get_word2vec_sent_embeddings
from author_classification_model_sparse import FFNN, train_model, show_confusion_matrix, evaluate_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch
import torch.nn as nn 

print("start")
  
path_to_train_file= "/Users/sarahannauffelmann/desktop/01_WiSe23:24/Seminar Klassifizierung/Projekt FFNN/letters/classifier_data_train.jsonl"
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
label_enc = LabelEncoder()
labels_encoded_train = label_enc.fit_transform(train_lang_labels)
labels_encoded_test = label_enc.transform(test_lang_labels)
target_names = label_enc.classes_   # ['da' 'de' 'en' 'fr' 'hu' 'it' 'sv']
#print(target_names)

y_train = torch.tensor(labels_encoded_train)
y_test = torch.tensor(labels_encoded_test)

# create Dataloader
dataloader_train = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=64)
dataloader_test = DataLoader(list(zip(X_test, y_test)), shuffle=True, batch_size=64)

input_size = word2vec_size
output_size = 7         #  7 different languages
hidden1 = 512
hidden2 = 512

model = FFNN(input_size, hidden1, hidden2, output_size)
model_name = "FFNN_sparse_lang"

# training parameters
num_epochs = 15
learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# # training loop
for epoch in range(num_epochs):
     train_loss = train_model(dataloader_train, model, optimizer, loss_func)
     print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}')

accuracy, gold_labels, predictions = evaluate_model(model, dataloader_test)

#print(set(gold_labels), set(predictions))
test_target_names = [lang for i, lang in enumerate(target_names) if i in set(gold_labels)]
print(test_target_names)

report = classification_report(gold_labels, predictions, target_names=test_target_names)
print(report)

show_confusion_matrix(gold_labels, predictions, test_target_names, model_name)