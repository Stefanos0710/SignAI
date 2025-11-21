"""
Train script for sign classification using full frame sequences.
Supports: variable sequence lengths, Masking, BiLSTM, Tokenizer gloss labels.
"""

import os
import csv
import io
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
import concurrent.futures

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Masking, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from typing import List, Dict, Tuple, Union

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

EXPECTED_FEATURES = 1086
CACHE_NAME = ".parsed_cache.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------------------
# PARSER – reads all frames of a CSV file
# ---------------------------------------------------------------------

def parse_csv_file(file_name: str, text: str) -> Tuple[dict, None]:
    """
    Parse a CSV text into a sample with all frames included.
    Returns:
        sample = {
            "keypoints_sequence": ndarray [frames, features],
            "gloss": str,
            "source_file": str
        }
    """
    lines = text.splitlines()
    rows = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            vals = np.fromstring(line, sep=",", dtype=np.float32)
        except:
            continue

        if vals.size >= EXPECTED_FEATURES:
            rows.append(vals[:EXPECTED_FEATURES])

    if len(rows) == 0:
        return None, ("no-valid-frames", file_name)

    arr = np.vstack(rows)

    sample = {
        "keypoints_sequence": arr.astype(np.float32),
        "gloss": os.path.splitext(file_name)[0],
        "source_file": file_name,
    }
    return sample, None

# ---------------------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------------------

def load_data(folder: str) -> List[dict]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(folder)

    cache_path = os.path.join(folder, CACHE_NAME)

    # Load cached parsed dataset if available
    if os.path.exists(cache_path):
        logging.info(f"Loading cached parsed dataset: {cache_path}")
        return pickle.load(open(cache_path, "rb"))

    logging.info("Parsing training dataset…")
    samples = []
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    file_texts = []
    for f in csv_files:
        path = os.path.join(folder, f)
        with open(path, "r", errors="replace") as tx:
            file_texts.append((f, tx.read()))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        future_map = {ex.submit(parse_csv_file, fn, txt): fn for fn, txt in file_texts}

        for fut in concurrent.futures.as_completed(future_map):
            result, err = fut.result()
            if result is not None:
                samples.append(result)

    pickle.dump(samples, open(cache_path, "wb"))
    logging.info(f"Parsed {len(samples)} samples total.")

    return samples

# ---------------------------------------------------------------------
# BUILD MODEL
# ---------------------------------------------------------------------

def build_model(max_frames, feature_dim, num_classes):
    """
    BiLSTM sign classification model
    """
    inputs = Input(shape=(max_frames, feature_dim), name="frames")

    x = Masking(mask_value=0.0)(inputs)

    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(128))(x)
    x = Dropout(0.3)(x)

    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(0.0004),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model

# ---------------------------------------------------------------------
# MAIN TRAINING
# ---------------------------------------------------------------------

if __name__ == "__main__":
    data_folder = "data/train_data"

    # 1. LOAD ALL SAMPLES
    samples = load_data(data_folder)
    logging.info("Building dataset…")

    # 2. Extract sequences + gloss labels
    sequences = []
    labels = []

    for s in samples:
        sequences.append(s["keypoints_sequence"])
        labels.append(s["gloss"])
        print(s["gloss"])  # print the gloss value (append() returns None, avoid printing that)

    # 3. Find longest sequence length
    max_len = max(seq.shape[0] for seq in sequences)
    feature_dim = sequences[0].shape[1]

    logging.info(f"Max sequence length: {max_len}")
    logging.info(f"Feature dimension: {feature_dim}")

    # 4. Pad sequences
    X = np.zeros((len(sequences), max_len, feature_dim), dtype=np.float32)

    for i, seq in enumerate(sequences):
        X[i, :seq.shape[0], :] = seq

    # 5. Tokenize gloss labels
    unique_labels = sorted(list(set(labels)))
    label_index = {lab: i for i, lab in enumerate(unique_labels)}

    y = np.array([label_index[lab] for lab in labels])
    y_cat = to_categorical(y, num_classes=len(unique_labels))

    # 6. Build model
    model = build_model(max_len, feature_dim, len(unique_labels))

    # 7. Train
    model.fit(
        X, y_cat,
        epochs=25,
        batch_size=8,
        validation_split=0.1,
        shuffle=True
    )

    # 8. Save model + label index
    model.save("sign_classifier.h5")
    json.dump(label_index, open("labels.json", "w"))

    logging.info("Training finished successfully.")
