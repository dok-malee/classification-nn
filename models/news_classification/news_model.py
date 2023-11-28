import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import json
from text_processing import create_sparse_vectors


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    assert len(data) > 0, "No data loaded"
    return data


def create_labels(loaded_data):
    labels = [entry['category'] for entry in loaded_data]
    label_encoder = LabelEncoder()

    # fits the encoder to the unique labels in your data and transforms the labels into numerical values
    # each color is assigned an integer
    y = label_encoder.fit_transform(labels)

    # one hot encoding for nominal data
    y_onehot = torch.nn.functional.one_hot(torch.tensor(y))

    label_mapping = dict(zip(label_encoder.classes_, y_onehot.numpy()))
    return y_onehot, label_mapping, len(label_encoder.classes_)


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
    path = '../../data/classification_data/data/news/classification/classification_news_train.jsonl'
    path_testset = '../../data/classification_data/data/news/classification/classification_news_eval.jsonl'

    news_data = load_data(file_path=path)
    #news_data_test = load_data(file_path=path_testset)

    X_sparse_train = create_sparse_vectors(news_data)
    print(X_sparse_train)

    labels_onehot, labels_mapping, count_labels = create_labels(news_data)
    print(labels_mapping)

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
    train_dataset = TensorDataset(X_train_tensor, labels_onehot)
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
