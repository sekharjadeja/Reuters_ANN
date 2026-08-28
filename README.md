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

---<img width="716" height="143" alt="dataset_contents" src="https://github.com/user-attachments/assets/0de1abe0-cd30-4469-adf4-f98252ad4ecc" />

<img width="1200" height="400" alt="training_results" src="https://github.com/user-attachments/assets/1d84403b-3bef-423a-9a25-5a102f5f5f67" />

## 📊 Model Performance & Results

* **Training Set:** 7,769 documents
* **Test Set:** 3,019 documents
* **Total Target Classes:** 90 unique financial topics
* **Exact-Match Test Accuracy:** `85.76%`
* **Test AUC Score:** `0.9229`








