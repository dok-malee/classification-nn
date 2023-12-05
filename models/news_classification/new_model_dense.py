# Load and preprocess data
from models.news_classification.text_processing import load_datasets, create_dense_embeddings

train_file = '../../data/classification_data/data/news/classification/classification_news_train.jsonl'
test_file = '../../data/classification_data/data/news/classification/classification_news_eval.jsonl'

train_docs, test_docs = load_datasets(train_file, test_file)

# Create dense embeddings using BERT
dense_embeddings_train, dense_embeddings_test, y_train, y_test = create_dense_embeddings(train_data=train_docs, test_data=test_docs)

# Print shape of dense embeddings
print(f"Train Dense Embeddings Shape: {dense_embeddings_train.shape}")
print(f"Test Dense Embeddings Shape: {dense_embeddings_test.shape}")
