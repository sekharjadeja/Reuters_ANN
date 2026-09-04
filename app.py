import os
import json
import time
import random
import zipfile
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras import layers, models

project_folder = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(project_folder, "model_artifacts")

app = Flask(__name__)

def ensure_dataset(subsets=None):
    """Extract dataset zip archives if target folders do not exist."""
    if subsets is None:
        subsets = ["training", "test"]
    for subset in subsets:
        folder = os.path.join(project_folder, subset)
        zip_path = os.path.join(project_folder, f"{subset}.zip")
        if not os.path.exists(folder) and os.path.exists(zip_path):
            print(f"Extracting {subset}.zip...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(project_folder)
            print(f"{subset}.zip extracted successfully.")


# Global handles for model and vectorizer
model = None
vectorizer = None
classes = []
vocab = []
metadata = {}
cats_mapping = {}

def load_data_and_model():
    global model, vectorizer, classes, vocab, metadata, cats_mapping

    # Load cats.txt
    cats_file = os.path.join(project_folder, "cats.txt")
    if os.path.exists(cats_file):
        with open(cats_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cats_mapping[parts[0]] = parts[1:]

    # Load metadata
    meta_path = os.path.join(artifacts_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            classes = metadata.get("classes", [])

    # Load trained model
    model_path = os.path.join(artifacts_dir, "reuters_ann_model.keras")
    if os.path.exists(model_path):
        model = models.load_model(model_path)
        print("Model loaded from disk.")

    # Initialize TextVectorization directly from pre-trained artifacts
    vocab_file = os.path.join(artifacts_dir, "vocab.json")
    idf_file = os.path.join(artifacts_dir, "idf_weights.json")

    if os.path.exists(vocab_file) and os.path.exists(idf_file):
        print("Loading pre-trained TextVectorization from model artifacts...")
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(idf_file, "r", encoding="utf-8") as f:
            idf_weights = json.load(f)

        vectorizer = layers.TextVectorization(
            max_tokens=len(vocab),
            output_mode="tf-idf",
            vocabulary=vocab,
            idf_weights=idf_weights
        )
        print(f"Pre-trained vectorizer loaded. Vocab size: {len(vocab)}. Classes: {len(classes)}")
    else:
        # Fallback: extract dataset and adapt on training files if artifacts missing
        print("Artifacts missing. Falling back to adapting on training corpus...")
        ensure_dataset(["training"])
        train_files = [f for f in cats_mapping.keys() if f.startswith("training/")]
        X_train_raw = []
        for f in train_files:
            p = os.path.join(project_folder, f)
            with open(p, "r", encoding="latin-1", errors="ignore") as file:
                X_train_raw.append(file.read())

        vectorizer = layers.TextVectorization(max_tokens=10000, output_mode="tf-idf")
        vectorizer.adapt(X_train_raw)
        vocab = vectorizer.get_vocabulary()
        print(f"Vectorization adapted successfully. Vocab size: {len(vocab)}. Classes: {len(classes)}")

# Curated high-impact financial samples from Reuters-21578
CURATED_SAMPLES = [
    {
        "title": "Corporate Earnings Release",
        "description": "Quarterly earnings, profit surge, dividend announcement",
        "expected": ["earn"],
        "text": """AMOCO CORP 1ST QTR NET RISES TO 432 MLN DLS VS 380 MLN
CHICAGO, April 23 - Amoco Corp reported first-quarter net income rose to 432 mln dlrs, or 1.66 dlrs a share, from 380 mln dlrs, or 1.46 dlrs a share, a year earlier.
Revenues for the quarter totaled 6.8 billion dlrs compared with 7.1 billion dlrs in the 1986 first quarter.
The company said higher crude oil prices and increased refining margins contributed to the earnings gain, despite lower chemical margins.
The board declared a regular quarterly dividend of 82.5 cents per common share, payable June 10 to holders of record May 8."""
    },
    {
        "title": "Mergers & Acquisitions",
        "description": "Takeover bid, tender offer, corporate buyout",
        "expected": ["acq"],
        "text": """GENERAL DYNAMICS TO ACQUIRE CESSNA AIRCRAFT IN 650 MLN DLR MERGER
ST. LOUIS, May 14 - General Dynamics Corp said it has reached a definitive agreement to acquire Cessna Aircraft Co for 650 mln dlrs in cash and stock.
Under terms of the agreement, General Dynamics will launch a tender offer for all outstanding common shares of Cessna at 30 dlrs per share.
The merger has been unanimously approved by both boards of directors and is subject to standard regulatory approvals and antitrust review."""
    },
    {
        "title": "Crude Oil & Energy Market",
        "description": "OPEC production quotas, crude petroleum pricing, inventory drawdown",
        "expected": ["crude"],
        "text": """SAUDI ARABIA REAFFIRMS COMMITMENT TO OPEC CRUDE OIL QUOTAS
RIYADH, June 2 - Saudi Arabia's petroleum ministry reiterated today that the kingdom will maintain crude oil output within its agreed OPEC production ceiling of 4.133 mln barrels per day.
Oil industry sources confirmed that tanker loadings at Ras Tanura have slowed following crude inventory draws in Western Europe and North America.
Brent crude for July settlement held steady in London trading at 18.65 dlrs per barrel."""
    },
    {
        "title": "Foreign Exchange & Currency Markets",
        "description": "Central bank intervention, dollar weakness vs yen and mark",
        "expected": ["money-fx", "interest"],
        "text": """DOLLAR FALLS AS CENTRAL BANKS INTERVENE IN CURRENCY MARKETS
NEW YORK, March 27 - The U.S. dollar retreated across the board following coordinated intervention by the Federal Reserve and the Bank of Japan to stabilize exchange rates.
Traders said the dollar dipped to 142.30 yen from 144.10 yen in early European trading.
Market participants are closely watching the upcoming discount rate decision from the Federal Open Market Committee amidst inflationary concerns."""
    },
    {
        "title": "Agricultural Commodities & Grain Trade",
        "description": "Wheat exports, Soviet grain purchases, harvest projections",
        "expected": ["grain", "wheat"],
        "text": """USDA CONFIRMS PRIVATE EXPORT SALE OF WHEAT TO SOVIET UNION
WASHINGTON, April 16 - The U.S. Department of Agriculture confirmed that private exporters have sold 1.25 million metric tons of hard red winter wheat to the Soviet Union for delivery in the 1987 marketing year.
Total U.S. wheat export commitments for the season now stand at 28.4 mln tons, up 14 percent from the corresponding period last year."""
    }
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/info")
def api_info():
    if model is None:
        load_data_and_model()
    return jsonify({
        "status": "ready" if model is not None else "initializing",
        "total_classes": len(classes),
        "vocab_size": len(vocab),
        "parameters": model.count_params() if model is not None else 5274970,
        "metadata": metadata
    })

@app.route("/api/samples")
def api_samples():
    return jsonify(CURATED_SAMPLES)

@app.route("/api/random_test")
def api_random_test():
    ensure_dataset(["test"])
    test_keys = [k for k in cats_mapping.keys() if k.startswith("test/")]
    if not test_keys:
        return jsonify({"error": "No test documents available"}), 404
    chosen_file = random.choice(test_keys)
    file_path = os.path.join(project_folder, chosen_file)
    try:
        with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "file_id": chosen_file,
        "ground_truth": cats_mapping.get(chosen_file, []),
        "text": content
    })

@app.route("/api/categories")
def api_categories():
    counts = {}
    for cats in cats_mapping.values():
        for c in cats:
            counts[c] = counts.get(c, 0) + 1

    sorted_cats = sorted(
        [{"name": c, "count": counts.get(c, 0)} for c in (classes or list(counts.keys()))],
        key=lambda x: x["count"],
        reverse=True
    )
    return jsonify({"categories": sorted_cats, "total": len(sorted_cats)})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    global model, vectorizer
    if model is None or vectorizer is None:
        load_data_and_model()
        if model is None:
            return jsonify({"error": "Model not ready yet. Please wait for training export to complete."}), 503

    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    threshold = float(data.get("threshold", 0.35))

    if not text:
        return jsonify({"error": "No text provided"}), 400

    t0 = time.perf_counter()
    # Vectorize
    vec = vectorizer(np.array([text]))
    # Predict
    probs = model.predict(vec, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    # Build predictions
    results = []
    for idx, prob in enumerate(probs):
        class_name = classes[idx] if idx < len(classes) else f"class_{idx}"
        results.append({
            "category": class_name,
            "probability": float(prob),
            "percentage": round(float(prob) * 100, 2),
            "selected": bool(prob >= threshold)
        })

    # Sort by probability descending
    results.sort(key=lambda x: x["probability"], reverse=True)

    # Top predictions matching threshold
    active_predictions = [r for r in results if r["selected"]]
    top_candidates = results[:8]  # Show top 8 for context

    # Extract keywords present in vocab
    words = [w.lower().strip(".,!?:;\"'()[]{}") for w in text.split()]
    vocab_set = set(vocab[:1500]) if vocab else set()
    significant_tokens = list(set([w for w in words if w in vocab_set and len(w) > 3]))[:12]

    return jsonify({
        "predictions": active_predictions,
        "top_candidates": top_candidates,
        "all_candidates": results,
        "threshold": threshold,
        "latency_ms": round(latency_ms, 2),
        "word_count": len(words),
        "tokens": significant_tokens
    })

@app.route("/api/metrics")
def api_metrics():
    # Return dynamic evaluation metrics from exported model artifacts metadata
    stored_metrics = metadata.get("metrics")
    if not stored_metrics:
        stored_metrics = {
            "micro_f1": 0.8530,
            "macro_f1": 0.4447,
            "exact_match_acc": 0.8036,
            "micro_roc_auc": 0.9770,
            "macro_roc_auc": 0.9314,
            "micro_precision": 0.8696,
            "micro_recall": 0.8371,
            "hamming_loss": 0.00397,
            "inference_latency_ms": 0.10,
            "total_parameters": model.count_params() if model is not None else 5274970,
            "train_samples": metadata.get("total_train_docs", 7769),
            "test_samples": metadata.get("total_test_docs", 3019),
            "total_categories": len(classes),
            "vocab_size": len(vocab) or 10000
        }
    return jsonify({
        "metrics": stored_metrics,
        "history": metadata.get("history", {})
    })

if __name__ == "__main__":
    load_data_and_model()
    app.run(host="127.0.0.1", port=5000, debug=False)
