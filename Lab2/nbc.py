import os
import re
import json
import string
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"# TensorFlow: suppress INFO & WARNING & ERROR messages
warnings.filterwarnings("ignore")# Suppress all Python warnings (including Keras deprecations)


# --- Global Config ---
BOOK_DIR = "harry_potter"
PAGE_SIZE = 400
BASE_EMBEDDING_DIM = 50
EMBEDDING_DIM = BASE_EMBEDDING_DIM + 1
NUM_CLASSES = 7
LAB1_MODEL_PATH = "word2vec_model_A.pth"
LAB1_VOCAB_PATH = "word_to_idx_A.json"
NUM_FILTERS = 100
KERNEL_SIZE = 3
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 64
DROPOUT_PROB = 0.5
WEIGHT_DECAY = 1e-4
MAX_WORDS = 20000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- System will be using device: {device} ---")

def clean_text(text: str) -> str:
    """Lowercase + remove punctuation/non-alphabetic chars."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ==========================================================
# --- Part A: PyTorch CNN Classifier ---
# ==========================================================
def prepare_dataset(book_dir, page_size):
    print("Starting data prep...")
    all_pages, all_labels = [], []

    for i in range(1, 8):
        file_path = os.path.join(book_dir, f"HP{i}.txt")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except FileNotFoundError:
            print(f"Cannot find file: {file_path}")
            continue
        print(f"Processing {file_path}...")

        clean_text = full_text.lower()
        punctuation_list = string.punctuation + '“”’'
        translator = str.maketrans('', '', punctuation_list)
        clean_text = clean_text.translate(translator)
        tokens = clean_text.split()

        num_pages_in_book = len(tokens) // page_size
        for j in range(num_pages_in_book):
            start, end = j * page_size, (j + 1) * page_size
            all_pages.append(tokens[start:end])
            all_labels.append(i - 1)  # labels from 0 to 6

    print("Data preparation complete.")
    return all_pages, all_labels

def embed_pages(pages, embedding_matrix, word_to_idx, page_size, base_embedding_dim):
    print("\nEmbedding pages with Lab 1 encoder and adding UNK flag...")
    new_embedding_dim = base_embedding_dim + 1
    embedded_pages = np.zeros((len(pages), page_size, new_embedding_dim))

    for i, page in enumerate(pages):
        for j, word in enumerate(page):
            word_index = word_to_idx.get(word)
            if word_index is not None:
                embedding_vector = embedding_matrix[word_index]
                embedded_pages[i, j, :base_embedding_dim] = embedding_vector
            else:
                embedded_pages[i, j, -1] = 1
    print("Embedding complete.")
    return embedded_pages

class CNNClassifier(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, num_classes, dropout_prob):
        super(CNNClassifier, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(2)
        x = self.dropout(x)
        return self.fc(x)

def predict_random_page(model, val_data, val_labels):
    model.eval()
    random_idx = random.randint(0, len(val_data) - 1)
    sample_page = val_data[random_idx]
    true_label_idx = val_labels[random_idx].item()
    sample_page_batch = sample_page.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sample_page_batch)
        _, predicted_idx = torch.max(output.data, 1)
        predicted_idx = predicted_idx.item()
    print("============================================================================")
    print("\n--- Single Page Prediction randomly selected---")
    print(f"Random page index: {random_idx}")
    print(f"True: Book {true_label_idx + 1}, Predicted: Book {predicted_idx + 1}")
    print("PREDICTION RESULT:", "Correct ✅" if true_label_idx == predicted_idx else "Incorrect ❌")
    print("============================================================================")

def run_MainTopic():
    pages, labels = prepare_dataset(BOOK_DIR, PAGE_SIZE)

    if not pages:
        print("No pages prepared for Part A.")
        return

    try:
        lab1_model_state = torch.load(LAB1_MODEL_PATH, map_location=torch.device('cpu'))
        embedding_matrix = lab1_model_state['embeddings.weight'].numpy()
        with open(LAB1_VOCAB_PATH, 'r') as f:
            word_to_idx = json.load(f)

        X = embed_pages(pages, embedding_matrix, word_to_idx, PAGE_SIZE, BASE_EMBEDDING_DIM)
        y = np.array(labels)
        print(f"\nFinal shape X: {X.shape}, y: {y.shape}")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=BATCH_SIZE, shuffle=False)

        print("\n--- Training Base CNN Classifier ---")
        model = CNNClassifier(EMBEDDING_DIM, NUM_FILTERS, KERNEL_SIZE, NUM_CLASSES, DROPOUT_PROB).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Val Acc: {accuracy:.2f}%")

        predict_random_page(model, X_val_tensor, y_val_tensor)

    except FileNotFoundError:
        print(f"Missing Lab 1 model ({LAB1_MODEL_PATH}) or vocab ({LAB1_VOCAB_PATH}).")
    except Exception as e:
        print(f"Error during embedding/training: {e}")

# ==========================================================
# --- Part B: SubTopic : How does increasing the width of the kernels affect the performance of the model? ---
# ==========================================================
def run_SubTopic():
    print("\n--- Running Part B now... ---")
    print("\n--- SubTopic : How does increasing the width of the kernels affect the performance of the model?---")
    texts, labels = [], []
    for i in range(1, 8):
        file_path = os.path.join(BOOK_DIR, f"HP{i}.txt")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except FileNotFoundError:
            continue
        tokens = clean_text(full_text).split()
        for start in range(0, len(tokens), PAGE_SIZE):
            page = tokens[start:start + PAGE_SIZE]
            if len(page) == PAGE_SIZE:
                texts.append(" ".join(page))
                labels.append(f"HP{i}")

    if not texts:
        print("No pages prepared for Part B.")
        return

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=PAGE_SIZE, padding='post', truncating='post')

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    y_cat = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(data, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

    results = {}
    for kernel_size in range(2, 6):
        print(f"\nTraining Keras CNN with kernel size = {kernel_size}")
        model = Sequential([
            Embedding(MAX_WORDS, BASE_EMBEDDING_DIM),# input_length=PAGE_SIZE),
            Conv1D(filters=128, kernel_size=kernel_size, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=EPOCHS//5, batch_size=64, validation_split=0.1, verbose=0)
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        results[kernel_size] = acc

    print("\nKernel Size vs Accuracy (Keras CNN):")
    for k, v in results.items():
        print(f"Kernel {k}: {v*100:.2f}%")

    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel("Kernel Size")
    plt.ylabel("Test Accuracy")
    plt.title("Effect of Kernel Size on CNN Accuracy (Keras)")
    #plt.show()
    plt.savefig("kernel_size_accuracy.png")
    print("Plot saved as kernel_size_accuracy.png")


if __name__ == "__main__":
    run_MainTopic()
    run_SubTopic()
