import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import json
from text_processing import create_sparse_vectors

path = '../../data/classification_data/data/news/classification/classification_news_train.jsonl'
path_testset = '../../data/classification_data/data/news/classification/classification_news_eval.jsonl'


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    assert len(data) > 0, "No data loaded"
    return data


def create_labels(loaded_data, label_encoder=None):
    labels = [entry['category'] for entry in loaded_data]
    if label_encoder is None:
        label_encoder = LabelEncoder()
        # fits the encoder to the unique labels in your data and transforms the labels into numerical values
        # each color is assigned an integer
        y = label_encoder.fit_transform(labels)
    else:
        # Use the existing label encoder for the test data
        y = label_encoder.transform(labels)

    # one hot encoding for nominal data

    label_mapping = dict(zip(label_encoder.classes_, y.numpy()))
    return y, label_mapping, len(label_encoder.classes_), label_encoder


class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedforwardNN, self).__init__()
        self.fully_connected_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fully_connected_2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fully_connected_1(x)  # Linear transformation with weights and biases
        x = self.relu(x)
        x = self.fully_connected_2(x)
        return x


if __name__ == "__main__":
    news_data = load_data(file_path=path)

    X_sparse_train, vectorizer = create_sparse_vectors(news_data)
    #print(X_sparse_train)

    y, labels_mapping, count_labels, label_encoder = create_labels(news_data)
    #print(labels_mapping)

    input_size = X_sparse_train.shape[1]
    hidden_size = 64
    output_size = count_labels

    # Instantiate the model
    model = FeedforwardNN(input_size, hidden_size, output_size)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 10
    batch_size = 64

    # Convert sparse matrix to PyTorch tensor
    X_train_tensor = torch.tensor(X_sparse_train.toarray(), dtype=torch.float32)
    # Create a DataLoader for training
    train_dataset = TensorDataset(X_train_tensor, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_x)

            loss = criterion(outputs, torch.argmax(batch_y, dim=1))
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), 'genre_classification_model.pth')

    news_data_test = load_data(file_path=path_testset)

    X_sparse_test, _ = create_sparse_vectors(news_data_test, text_column='headline', vectorizer=vectorizer)
    X_test_tensor = torch.tensor(X_sparse_test.toarray(), dtype=torch.float32)

    y, labels_mapping, count_labels, _ = create_labels(news_data_test, label_encoder=label_encoder)
    # print(labels_mapping)

    # Load the trained model
    model = FeedforwardNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('genre_classification_model.pth'))
    model.eval()

    # Perform forward pass on the test set
    with torch.no_grad():
        outputs_test = model(X_test_tensor)

    # Convert outputs to class predictions
    predictions = torch.argmax(outputs_test, dim=1).numpy()

    # Convert one-hot encoded labels to class indices
    true_labels = torch.argmax(y, dim=1).numpy()

    # Print classification report
    report = classification_report(true_labels, predictions)
    print("Classification Report:\n", report)
