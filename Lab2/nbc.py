# --------------------
# 0. Imports
# --------------------
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --------------------
# 1. Load and preprocess text
# --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

books_dir = "harry_potter"  # folder containing HP1.txt ... HP7.txt
page_size = 300  # words per page
texts = []
labels = []

for i in range(1, 8):
    file_path = os.path.join(books_dir, f"HP{i}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        words = clean_text(f.read()).split()
        for start in range(0, len(words), page_size):
            page_words = words[start:start + page_size]
            if len(page_words) == page_size:
                texts.append(" ".join(page_words))
                labels.append(f"HP{i}")

print(f"Loaded {len(texts)} pages from 7 books.")

# --------------------
# 2. Tokenize text
# --------------------
max_words = 20000
max_len = page_size

tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y_cat = to_categorical(y)

# --------------------
# 3. Train-test split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

# --------------------
# 4. Loop over kernel sizes and train/evaluate
# --------------------
embedding_dim = 100
results = {}

for kernel_size in range(2, 6):  # 2,3,4,5
    print(f"\n==============================")
    print(f"Training CNN with kernel size = {kernel_size}")
    print(f"==============================\n")
    
    # Build CNN model
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(Conv1D(filters=128, kernel_size=kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    results[kernel_size] = acc
    #print(f" Test Accuracy for kernel size {kernel_size}: {acc*100:.2f}%")

# --------------------
# 5. Predict on a specific page
# --------------------
def predict_page(page_number, X_data, y_data, model, label_encoder):
    if page_number < 0 or page_number >= len(X_data):
        print(f" Page number {page_number} is out of range (0 to {len(X_data)-1}).")
        return
    
    sample_page = X_data[page_number]
    pred = model.predict(np.expand_dims(sample_page, axis=0), verbose=0)
    pred_label = label_encoder.inverse_transform([np.argmax(pred)])[0]
    true_label = label_encoder.inverse_transform([np.argmax(y_data[page_number])])[0]

    print(f"\n Page {page_number}")
    print("Predicted:", pred_label)
    print("True:     ", true_label)

# Prompt user for page number
print(f"\nThere are {len(X_test)} test pages available.")
page_num = int(input(f"Enter a page number to test (0 to {len(X_test)-1}): "))
# Use the last trained model (kernel size = 5) for prediction
predict_page(page_num, X_test, y_test, model, label_encoder)

# --------------------
# 6. Summary
# --------------------
print("\n==============================")
print("Kernel Size vs Accuracy Summary:")
for k, v in results.items():
    print(f"Kernel size {k}: Accuracy = {v*100:.2f}%")
print("==============================\n")
