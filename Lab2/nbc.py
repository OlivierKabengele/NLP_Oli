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
from tensorflow.keras.initializers import Constant

import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# =========================
# --- Global Config ---
# =========================
BOOK_DIR = "harry_potter"
PAGE_SIZE = 400
BASE_EMBEDDING_DIM = 50
EMBEDDING_DIM = BASE_EMBEDDING_DIM + 1  # last dimension reserved for UNK flag
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
print(f"--- Using device: {device} ---")

# =========================
# --- Utility Functions ---
# =========================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def load_lab1_embeddings(model_path, vocab_path):
    """Load Lab1 pretrained embeddings and vocabulary."""
    lab1_state = torch.load(model_path, map_location="cpu")
    embedding_matrix = lab1_state['embeddings.weight'].numpy()
    with open(vocab_path, 'r') as f:
        word_to_idx = json.load(f)
    return embedding_matrix, word_to_idx

# =========================
# --- Part A: PyTorch CNN ---
# =========================
def prepare_dataset(book_dir, page_size):
    all_pages, all_labels = [], []
    for i in range(1, 8):
        file_path = os.path.join(book_dir, f"HP{i}.txt")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except FileNotFoundError:
            print(f"Cannot find file: {file_path}")
            continue

        punctuation_list = string.punctuation + '“”’'
        translator = str.maketrans('', '', punctuation_list)
        tokens = full_text.lower().translate(translator).split()

        num_pages = len(tokens) // page_size
        for j in range(num_pages):
            start, end = j * page_size, (j + 1) * page_size
            all_pages.append(tokens[start:end])
            all_labels.append(i - 1)  # labels 0-6
    return all_pages, all_labels

def embed_pages(pages, embedding_matrix, word_to_idx, page_size, base_dim):
    """Embed pages using Lab1 embeddings and add UNK flag."""
    embedded_pages = np.zeros((len(pages), page_size, base_dim + 1))
    for i, page in enumerate(pages):
        for j, word in enumerate(page):
            idx = word_to_idx.get(word)
            if idx is not None:
                embedded_pages[i, j, :base_dim] = embedding_matrix[idx]
            else:
                embedded_pages[i, j, -1] = 1  # UNK flag
    return embedded_pages

def build_merged_vocab(pages_tokens, lab1_word_to_idx, max_new_words=None):
    """
    Keep Lab1 indices intact; append new words with new indices.
    Returns merged_word_to_idx and a list of new words in order.
    """
    merged = dict(lab1_word_to_idx)  # keep existing indices
    next_idx = max(merged.values()) + 1 if merged else 0
    new_words = []
    for page in pages_tokens:
        for w in page:
            if w not in merged:
                if (max_new_words is not None) and (len(new_words) >= max_new_words):
                    continue
                merged[w] = next_idx
                new_words.append(w)
                next_idx += 1
    # Add UNK if not present
    if "<unk>" not in merged:
        merged["<unk>"] = next_idx
    return merged, new_words

def pages_to_indices(pages_tokens, word_to_idx):
    unk_idx = word_to_idx.get("<unk>")
    idx_pages = []
    for page in pages_tokens:
        idx_pages.append([word_to_idx.get(w, unk_idx) for w in page])
    return np.array(idx_pages, dtype=np.int64)


def make_torch_embedding_from_lab1(embedding_matrix_lab1, lab1_vocab, merged_vocab, embed_dim, trainable=True):
    vocab_size = max(merged_vocab.values()) + 1
    weight = np.random.normal(scale=0.01, size=(vocab_size, embed_dim)).astype(np.float32)

    # copy Lab1 rows where available (clip in case Lab1 had larger dim—here they match)
    for w, lab1_idx in lab1_vocab.items():
        merged_idx = merged_vocab.get(w)
        if merged_idx is not None:
            weight[merged_idx, :embed_dim] = embedding_matrix_lab1[lab1_idx, :embed_dim]

    # Optionally zero-init UNK row
    unk_idx = merged_vocab.get("<unk>")
    if unk_idx is not None:
        weight[unk_idx] = 0.0

    emb = nn.Embedding(vocab_size, embed_dim)
    emb.weight.data = torch.tensor(weight)
    emb.weight.requires_grad = trainable
    return emb


class CNNClassifier(nn.Module):
    def __init__(self, embedding_layer, num_filters, kernel_size, num_classes, dropout_prob):
        super().__init__()
        self.embedding = embedding_layer
        self.conv1d = nn.Conv1d(in_channels=self.embedding.embedding_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x_idx):
        # x_idx: (batch, seq_len) of token indices
        x = self.embedding(x_idx)            # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)               # (batch, embed_dim, seq_len)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(2)
        x = self.dropout(x)
        return self.fc(x)


def run_part_a():
    pages_tokens, labels = prepare_dataset(BOOK_DIR, PAGE_SIZE)
    if not pages_tokens:
        print("No pages found for Part A.")
        return

    # Load Lab1
    lab1_embed, lab1_vocab = load_lab1_embeddings(LAB1_MODEL_PATH, LAB1_VOCAB_PATH)

    # Build merged vocab (Lab1 + new)
    merged_vocab, new_words = build_merged_vocab(pages_tokens, lab1_vocab)
    print("===========================================")
    print(f"Merged vocab size: {len(merged_vocab)} (new words: {len(new_words)})")
    print("===========================================")

    # Convert pages to indices
    X_idx = pages_to_indices(pages_tokens, merged_vocab)
    y = np.array(labels)

    # Split
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(X_idx, y, test_size=0.2, random_state=42, stratify=y)

    # Tensors
    X_train_t = torch.tensor(X_train_idx, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val_idx,   dtype=torch.long)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE, shuffle=False)

    # Embedding layer (pretrained+random)
    emb_layer = make_torch_embedding_from_lab1(
        embedding_matrix_lab1=lab1_embed,
        lab1_vocab=lab1_vocab,
        merged_vocab=merged_vocab,
        embed_dim=BASE_EMBEDDING_DIM,
        trainable=True,  # allow CNN to learn/fine-tune
    )

    model = CNNClassifier(
        embedding_layer=emb_layer,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        num_classes=NUM_CLASSES,
        dropout_prob=DROPOUT_PROB
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 20 == 0 or epoch == 0 or epoch == EPOCHS-1:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    preds = torch.argmax(model(bx), dim=1)
                    correct += (preds == by).sum().item()
                    total += by.size(0)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {100*correct/total:.2f}%")

    # Random page prediction now uses index tensors
    def predict_random_page_idx():
        model.eval()
        ridx = random.randint(0, len(X_val_t)-1)
        sample = X_val_t[ridx].unsqueeze(0).to(device)
        true_label = y_val_t[ridx].item()
        with torch.no_grad():
            pred = torch.argmax(model(sample), dim=1).item()
        print("============================================================================")
        print("\n--- Single Page Prediction ---")
        print(f"Random Page Index: {ridx}, True: Book {true_label+1}, Predicted: Book {pred+1}")
        print("Correct ✅" if true_label == pred else "Incorrect ❌")
        print("============================================================================")


    predict_random_page_idx()


# =========================
# --- Part B: SUBTOPIC CNN Kernel Size Experiment ---
# =========================
def run_part_b(pages_tokens, labels, merged_vocab, emb_layer):
    print("\n--- Running Part B: Kernel Size vs Accuracy  ---")

    # Convert pages to indices again
    X_idx = pages_to_indices(pages_tokens, merged_vocab)
    y = np.array(labels)

    X_train_idx, X_val_idx, y_train, y_val = train_test_split(X_idx, y, test_size=0.2, random_state=42, stratify=y)

    X_train_t = torch.tensor(X_train_idx, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val_idx,   dtype=torch.long)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE, shuffle=False)

    results = {}
    for kernel_size in range(2,7):
        print(f"\nTraining CNN with kernel size={kernel_size}")
        model = CNNClassifier(
            embedding_layer=emb_layer,
            num_filters=NUM_FILTERS,
            kernel_size=kernel_size,
            num_classes=NUM_CLASSES,
            dropout_prob=DROPOUT_PROB
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        for epoch in range(EPOCHS//5):  # fewer epochs for experiment
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = torch.argmax(model(bx), dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)
        acc = correct/total
        results[kernel_size] = acc
        print(f"Kernel {kernel_size}: Val Acc = {acc*100:.2f}%")

    # Plot results
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel("Kernel Size")
    plt.ylabel("Validation Accuracy")
    plt.title("Effect of Kernel Size on CNN Accuracy ")
    plt.savefig("kernel_size_accuracy.png")
    print("Plot saved as kernel_size_accuracy.png")


# =========================
# --- Main Runner ---
# =========================
if __name__ == "__main__":
    pages_tokens, labels = prepare_dataset(BOOK_DIR, PAGE_SIZE)
    lab1_embed, lab1_vocab = load_lab1_embeddings(LAB1_MODEL_PATH, LAB1_VOCAB_PATH)
    merged_vocab, new_words = build_merged_vocab(pages_tokens, lab1_vocab)

    emb_layer = make_torch_embedding_from_lab1(
        embedding_matrix_lab1=lab1_embed,
        lab1_vocab=lab1_vocab,
        merged_vocab=merged_vocab,
        embed_dim=BASE_EMBEDDING_DIM,
        trainable=True
    )

    run_part_a()  # original training
    run_part_b(pages_tokens, labels, merged_vocab, emb_layer)  # continuation
