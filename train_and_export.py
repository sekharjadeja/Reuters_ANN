import os
import json
import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, models

project_folder = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(project_folder, "model_artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

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
classes = list(mlb.classes_)
num_classes = len(classes)

# 3. Vectorize text data (TF-IDF)
MAX_WORDS = 10000
print("Adapting TextVectorization...")
vectorizer = layers.TextVectorization(max_tokens=MAX_WORDS, output_mode="tf-idf")
vectorizer.adapt(X_train_raw)

X_train = vectorizer(np.array(X_train_raw))
X_test = vectorizer(np.array(X_test_raw))

# 4. Build Model
print("Building neural network...")
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

# 5. Train Model
print("Training model for 10 epochs...")
t0 = time.time()
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)
train_time = time.time() - t0
print(f"Training completed in {train_time:.2f}s")

# 6. Save Model and Artifacts
print("Saving model and vectorizer vocabulary...")
model.save(os.path.join(artifacts_dir, "reuters_ann_model.keras"))

# Save vocabulary
vocab = vectorizer.get_vocabulary()
with open(os.path.join(artifacts_dir, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(vocab, f)

# Save classes and history
metadata = {
    "classes": classes,
    "num_classes": num_classes,
    "vocab_size": len(vocab),
    "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    "train_time": train_time,
    "total_train_docs": len(X_train_raw),
    "total_test_docs": len(X_test_raw)
}
with open(os.path.join(artifacts_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Export complete! All artifacts saved to:", artifacts_dir)
