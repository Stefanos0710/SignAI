import os
import re
import sys
import pickle
import logging
import concurrent.futures
from typing import Tuple, List, Dict

import numpy as np
import tensorflow as tf
import datetime
import json

from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Masking, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
# exclude face features; only pose + hands for higher accuracy on small datasets
EXPECTED_FEATURES = 150
CACHE_NAME = ".parsed_cache.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------------------
# PARSER – reads all frames of a CSV file
# ---------------------------------------------------------------------

def parse_csv_file(file_name: str, text: str) -> Tuple[dict | None, tuple | None]:
    """
    Parse a CSV text produced by the preprocessing step.
    Expects rows of the form: Video_Name, Gloss, Frame, pose_x,y..., hand_x,y..., face_x,y...,
    This parser will skip the face columns and only return pose+hand features.
    Returns a sample dict or (None, error).
    """
    import io
    import csv
    import re

    gloss_value = None
    rows = []

    try:
        rdr = csv.reader(io.StringIO(text))
        first = True
        for row in rdr:
            if not row:
                continue
            # detect header row
            if first:
                first = False
                if len(row) >= 2 and (row[0].lower().startswith("video") or row[1].lower().startswith("gloss")):
                    continue

            # Need at least Video_Name and Gloss
            if len(row) < 2:
                continue

            # capture gloss from second column if available
            if gloss_value is None and len(row) >= 2:
                gloss_value = row[1]

            start_idx = None
            if len(row) - 3 >= EXPECTED_FEATURES:
                # there is a frame column at index 2, but numeric data starts at 3
                start_idx = 3
            elif len(row) - 2 >= EXPECTED_FEATURES:
                # no Frame column, numeric data starts at 2
                start_idx = 2
            else:
                # no enough numeric columns in this row
                continue

            numeric_tokens = row[start_idx:]

            # convert token to floats
            vals_list = []
            for tok in numeric_tokens:
                try:
                    vals_list.append(float(tok))
                except Exception:
                    # stop conversion when if non-numeric encountered
                    break
                if len(vals_list) >= EXPECTED_FEATURES:
                    break

            if len(vals_list) >= EXPECTED_FEATURES:
                vals = np.array(vals_list[:EXPECTED_FEATURES], dtype=np.float32)
                rows.append(vals)

    except Exception:
        # if csv parse fails, will fall back below
        pass

    # if csv style parsing produced frames, return them
    if len(rows) > 0:
        arr = np.vstack(rows)
        sample = {
            "keypoints_sequence": arr.astype(np.float32),
            "gloss": gloss_value if gloss_value is not None else os.path.splitext(file_name)[0],
            "source_file": file_name,
        }
        logging.info(f"Parsed {arr.shape[0]} frames (pose+hand) from CSV: {file_name} (gloss={sample['gloss']})")
        return sample, None

    # --- Fallback: as last resort, extract numbers from file text and try to chunk into frames of EXPECTED_FEATURES ---
    try:
        toks = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)

        if len(toks) >= EXPECTED_FEATURES:
            arr = np.array(toks, dtype=np.float32)
            num_frames = arr.size // EXPECTED_FEATURES
            if num_frames > 0:
                arr = arr[: num_frames * EXPECTED_FEATURES]
                arr = arr.reshape((num_frames, EXPECTED_FEATURES))
                sample = {
                    "keypoints_sequence": arr.astype(np.float32),
                    "gloss": os.path.splitext(file_name)[0],
                    "source_file": file_name,
                }
                logging.info(f"Parsed (fallback) {num_frames} frames from: {file_name}")
                return sample, None
    except Exception as e:
        logging.debug(f"Fallback parse failed for {file_name}: {e}")

    return None, ("no-valid-frames", file_name)

# ---------------------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------------------
def load_data(folder: str) -> List[dict]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(folder)

    cache_path = os.path.join(folder, CACHE_NAME)

    if '--rebuild-cache' in sys.argv:
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logging.info(f"Removed existing cache due to --rebuild-cache: {cache_path}")
        except Exception as e:
            logging.warning(f"Could not remove cache file: {e}")

    if os.path.exists(cache_path):
        try:
            loaded = pickle.load(open(cache_path, "rb"))
            if isinstance(loaded, list) and len(loaded) > 0 and all(isinstance(s, dict) and 'keypoints_sequence' in s for s in loaded):
                logging.info(f"Loaded cached parsed dataset: {cache_path}")
                return loaded
            else:
                logging.warning(f"Cached parsed dataset is empty or malformed, will reparse: {cache_path}")
        except Exception as e:
            logging.warning(f"Failed to load cached parsed dataset ({cache_path}): {e}. Re-parsing CSV files.")

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

    try:
        pickle.dump(samples, open(cache_path, "wb"))
        logging.info(f"Wrote parsed cache to: {cache_path}")
    except Exception as e:
        logging.warning(f"Failed to write parsed cache: {e}")

    logging.info(f"Parsed {len(samples)} samples total.")
    return samples

# ---------------------------------------------------------------------
# Model Creation
# ---------------------------------------------------------------------
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

def build_model(max_frames, feature_dim, num_classes):
    """
    Smaller BiLSTM model suitable for small datasets.
    Adjusted for 150 features instead of 1087.
    """
    inputs = Input(shape=(max_frames, feature_dim), name="frames")
    x = Masking(mask_value=0.0)(inputs)

    # BiLSTM Layer 1
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    # BiLSTM Layer 2
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.2)(x)

    # Dense Layer
    x = Dense(64, activation="relu")(x)  # kleiner als vorher
    x = Dropout(0.2)(x)

    # Output Layer
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


# ---------------------------------------------------------------------
# Utility: stratified split that ensures each class is present in val when possible
# ---------------------------------------------------------------------
def stratified_split(labels: List[str], val_fraction: float=0.1, seed: int=42):
    """
    Returns train_idx, val_idx lists. Ensures every class with >=2 samples
    contributes at least one sample to validation. Classes with 1 sample stay in train.
    """
    from collections import defaultdict
    rng = np.random.RandomState(seed)
    label_to_indices = defaultdict(list)
    for i, lab in enumerate(labels):
        label_to_indices[lab].append(i)

    train_idx = []
    val_idx = []

    # give each class with >=2 samples one sample for val
    for lab, idxs in label_to_indices.items():
        if len(idxs) >= 2:
            idxs_sh = list(idxs)
            rng.shuffle(idxs_sh)
            # allocate one to val
            val_idx.append(idxs_sh.pop())
            train_idx.extend(idxs_sh)
        else:
            # only one sample — keep in train
            train_idx.extend(idxs)

    #  if val fraction is larger then add random samples from training set
    total = len(labels)
    desired_val = max(1, int(total * val_fraction))
    current_val = len(val_idx)
    if current_val < desired_val:
        remaining = list(train_idx)
        rng.shuffle(remaining)
        need = desired_val - current_val
        take = remaining[:need]
        # move these from train to val
        for t in take:
            train_idx.remove(t)
            val_idx.append(t)

    # final shuffle
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    logging.info(f"Stratified split -> Train: {len(train_idx)}, Val: {len(val_idx)} (requested val_fraction={val_fraction})")
    return train_idx, val_idx

# ---------------------------------------------------------------------
# MAIN TRAINING
# ---------------------------------------------------------------------
if __name__ == "__main__":
    data_folder = "data/train_data"

    samples = load_data(data_folder)

    if not samples:
        logging.error("No training samples found in '%s'.", data_folder)
        sys.exit(1)

    logging.info("Building dataset…")

    sequences = []
    labels = []
    for s in samples:
        sequences.append(s["keypoints_sequence"])
        labels.append(s["gloss"])

    # basic checks
    num_samples = len(sequences)
    unique_labels = sorted(list(set(labels)))
    num_classes = len(unique_labels)
    logging.info(f"Samples: {num_samples}, Unique labels: {num_classes}")

    # if  classes > samples, warn and exit
    if num_classes > num_samples:
        logging.warning("Number of classes > number of samples. Many classes have only one sample. Consider gathering more data.")
    # Find longest sequence length
    max_len = max(seq.shape[0] for seq in sequences)
    feature_dim = sequences[0].shape[1]
    logging.info(f"Max sequence length: {max_len}")
    logging.info(f"Feature dimension: {feature_dim}")

    # pad sequences
    X = np.zeros((num_samples, max_len, feature_dim), dtype=np.float32)
    for i, seq in enumerate(sequences):
        X[i, :seq.shape[0], :] = seq

    # tokenize labels
    label_index = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_index[lab] for lab in labels])
    y_cat = to_categorical(y, num_classes=num_classes)

    # startified split
    train_idx, val_idx = stratified_split(labels, val_fraction=0.1, seed=123)
    X_train, y_train = X[train_idx], y_cat[train_idx]
    X_val, y_val = X[val_idx], y_cat[val_idx]

    # if val set is empty (by small dataset like now) create a tiny val set from train (1 sample)
    if X_val.shape[0] == 0 and X_train.shape[0] > 1:
        logging.warning("Validation set is empty after stratification; moving 1 sample from train to val.")
        X_val = X_train[-1:].copy()
        y_val = y_train[-1:].copy()
        X_train = X_train[:-1]
        y_train = y_train[:-1]

    # build model
    model = build_model(max_len, feature_dim, num_classes)

    # class weights
    from collections import Counter
    counts = Counter(y)
    class_weight = {}
    # compute inverse frequency
    for k in range(num_classes):
        class_weight[k] = (num_samples / float(num_classes * (counts.get(k, 0) + 1e-6)))
    logging.info(f"Class weights: {class_weight}")

    # callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]

    # choose batch size sensibly
    batch_size = min(8, max(1, X_train.shape[0]))
    epochs = 30

    logging.info(f"Training with batch_size={batch_size}, epochs={epochs}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        class_weight=class_weight
    )

    # Ssave model + creation of model timestamped artifacts
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"sign_classifier_{ts}.keras"
    model_path = os.path.join(models_dir, model_filename)
    model.save(model_path)

    labels_path = os.path.join(models_dir, f"labels_{ts}.json")
    with open(labels_path, "w", encoding="utf-8") as lf:
        json.dump(label_index, lf, ensure_ascii=False)

    history_path = os.path.join(models_dir, f"history_{ts}.json")
    with open(history_path, "w", encoding="utf-8") as hf:
        json.dump(history.history, hf)

    logging.info(f"Saved model to: {model_path}")
    logging.info(f"Saved labels to: {labels_path}")
    logging.info(f"Saved history to: {history_path}")

    # try to plot training curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        acc = history.history.get('accuracy') or history.history.get('acc')
        val_acc = history.history.get('val_accuracy') or history.history.get('val_acc')
        loss = history.history.get('loss')
        val_loss = history.history.get('val_loss')

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        if acc is not None and val_acc is not None:
            axs[0].plot(acc, label='train_accuracy')
            axs[0].plot(val_acc, label='val_accuracy')
            axs[0].set_title('Accuracy')
            axs[0].legend()
        else:
            axs[0].text(0.5, 0.5, 'No accuracy data', ha='center')

        if loss is not None and val_loss is not None:
            axs[1].plot(loss, label='train_loss')
            axs[1].plot(val_loss, label='val_loss')
            axs[1].set_title('Loss')
            axs[1].legend()
        else:
            axs[1].text(0.5, 0.5, 'No loss data', ha='center')

        plt.tight_layout()
        plot_path = os.path.join(models_dir, f"training_{ts}.png")
        fig.savefig(plot_path)
        plt.close(fig)
        logging.info(f"Saved training plot to: {plot_path}")
    except Exception as e:
        logging.warning(f"Could not create training plot (matplotlib missing?): {e}")

    # try to create model
    try:
        symlink_model = os.path.join(models_dir, "latest_sign_classifier.keras")
        # remove existing quick-link
        if os.path.exists(symlink_model):
            try:
                os.remove(symlink_model)
            except Exception:
                pass
        # copy file
        import shutil
        shutil.copy2(model_path, symlink_model)
        shutil.copy2(labels_path, os.path.join(models_dir, "labels.json"))
    except Exception:
        pass

    logging.info("Training finished successfully.")
