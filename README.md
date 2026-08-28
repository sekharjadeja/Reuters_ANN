# Reuters-21578 Multi-Label Text Classification using ANN

An end-to-end Machine Learning and Natural Language Processing (NLP) pipeline built with **TensorFlow/Keras** to classify financial news articles from the benchmark **Reuters-21578** dataset into multiple topic categories.

---

## 📌 Project Overview
Financial news articles frequently cover overlapping subjects simultaneously (e.g., an article discussing both earnings reports and corporate acquisitions). To address this, the problem is modeled as a **multi-label text classification task** rather than single-label classification.

This repository contains a full pipeline that parses local raw document files, extracts TF-IDF feature representations (including word bigrams), and trains a Deep Artificial Neural Network (ANN) optimized with Batch Normalization, Dropout, and L2 Regularization.

---

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.12
* **Deep Learning Framework:** TensorFlow 2.x / Keras
* **NLP & Feature Engineering:** Scikit-Learn, NLTK
* **Data Manipulation:** NumPy

---

## ⚙️ Key Technical Features

1. **Custom Data Parsing:** Reads raw document files directly from directory structures and maps multi-topic labels via `cats.txt`.
2. **Advanced Vectorization:** Utilizes `tf.keras.layers.TextVectorization` with TF-IDF output mode and bigram extraction (`ngrams=(1, 2)`) to capture context-aware word pairs.
3. **Multi-Label Architecture:** Uses a Multi-Layer Perceptron (MLP) ending in a **Sigmoid** activation layer combined with **Binary Cross-Entropy Loss** to evaluate probability independently per topic.
4. **Regularization & Optimization:**
5. Features L2 kernel weight regularization, Dropout, Batch Normalization, and Early Stopping callbacks to prevent overfitting.


## 📊 Model Performance & Results

* **Contents of dataset:**
<img width="716" height="143" alt="dataset_contents" src="https://github.com/user-attachments/assets/0e3d0493-1d34-4c20-97df-91c472e3e512" />

* **training of ANN model with 10 epochs:**
<img width="1451" height="383" alt="training1" src="https://github.com/user-attachments/assets/914316d1-9d28-450a-bbea-0825e3355761" /> <img width="1491" height="402" alt="training2" src="https://github.com/user-attachments/assets/a1e9795b-d9e6-404a-b915-5005583204c2" />



* **Overall Metrics:**
<img width="690" height="557" alt="overall_metrics" src="https://github.com/user-attachments/assets/d278966a-eb16-4ff0-94e7-06aeb6b0d9bf" />



* **Model accuracy and Loss during training:**
    <img width="1200" height="400" alt="training_results" src="https://github.com/user-attachments/assets/03483211-11aa-407e-bd3f-74a5a476c657" />









