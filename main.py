import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, models

# Silence TensorFlow startup info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Point to project root directory
project_folder = os.path.dirname(os.path.abspath(__file__))

# 1. Parse categories mapping from cats.txt
cats_file_path = os.path.join(project_folder, "cats.txt")
file_to_categories = {}

with open(cats_file_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            file_id = parts[0]
            categories = parts[1:]
            file_to_categories[file_id] = categories

train_files = [f for f in file_to_categories if f.startswith("training/")]
test_files = [f for f in file_to_categories if f.startswith("test/")]

# Helper function to read raw article text
def read_text(file_id):
    path = os.path.join(project_folder, file_id)
    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        return f.read()

print("Reading articles from local files...")
X_train_raw = [read_text(f) for f in train_files]
y_train_raw = [file_to_categories[f] for f in train_files]

X_test_raw = [read_text(f) for f in test_files]
y_test_raw = [file_to_categories[f] for f in test_files]

# 2. Encode category labels into multi-label binary vectors
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train_raw)
y_test = mlb.transform(y_test_raw)

num_classes = y_train.shape[1]
print(f"Loaded {len(X_train_raw)} Training documents and {len(X_test_raw)} Test documents.")
print(f"Total Unique Classes: {num_classes}\n")

# 3. Vectorize text data (TF-IDF)
MAX_WORDS = 10000
vectorizer = layers.TextVectorization(max_tokens=MAX_WORDS, output_mode="tf-idf")
vectorizer.adapt(X_train_raw)

X_train = vectorizer(np.array(X_train_raw))
X_test = vectorizer(np.array(X_test_raw))

# 4. Build Neural Network Architecture
model = models.Sequential([
    layers.Input(shape=(MAX_WORDS,)),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 5. Train & Evaluate Model
print("Training Neural Network Model...")
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")




"""THE RESULTS AND ACCURACY OF THE ABOVE MODEL"""

"""
==================================================
              HISTORICAL RUN RESULTS
==================================================
Executed on: Local Virtual Environment (.venv)

Training Log:
Epoch 1/10  - loss: 0.0879 - accuracy: 0.5415 - val_loss: 0.0312 - val_accuracy: 0.7117
Epoch 5/10  - loss: 0.0072 - accuracy: 0.8929 - val_loss: 0.0202 - val_accuracy: 0.7941
Epoch 10/10 - loss: 0.0046 - accuracy: 0.9068 - val_loss: 0.0211 - val_accuracy: 0.7954

Test Evaluation (3,019 samples):
- Final Test Loss    : 0.0461
- Final Test Accuracy: 85.72%
==================================================
"""
