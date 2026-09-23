"""
1) Loading the dataset:
       - load_split(split: str)      -- read the npz files, keep crop JPEGs undecoded
       - create_dataset(split: str)  -- wrap load_split into a tf.data.Dataset that decodes
                                         each clip's crops lazily, per batch
2) Training:
       - build_dataset(split, ...)   -- pose heatmaps + cached hand/face DINOv3 features -> y,
                                         as a tf.data.Dataset (WordClassifier's input, not the
                                         raw crops create_dataset() decodes)
       - train_main(...)             -- compile WordClassifier, fit, save the best checkpoint
"""
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.models.model import WordClassifier  # noqa: E402
from signai.word_classification.models.hand_stream import load_hand_features  # noqa: E402
from signai.word_classification.models.face_stream import load_face_features  # noqa: E402
from signai.word_classification.paths import PROCESSED_DIR  # noqa: E402

DATASET_DIR = str(PROCESSED_DIR)
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

SPLITS = ["train", "val", "test"]
CROP_TYPES = ("left_hand", "right_hand", "mouth")

# training parameters
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

def load_split(split: str):
    """Load a dataset split (train/val/test) from dataset/processed/.

    Crops are returned as the raw JPEG byte strings stored in {split}_images.npz, not
    decoded to pixels.
    create_dataset() decodes lazily, one batch at a time, via tf.io.decode_jpeg.

    Returns:
        X: per-landmark Gaussian heatmaps, (N, 32, N_LANDMARKS, HEATMAP_RESOLUTION, HEATMAP_RESOLUTION)
            float32 (see preprocessing.py's keypoints_to_heatmaps)
        y: label ids, (N,) int64
        classes: label names, (num_classes,) str, classes[y[i]] == label of sample i
        crops: dict with "left_hand"/"right_hand"/"mouth", each (N, 32) object array of
            JPEG-encoded bytes
    """
    data = np.load(os.path.join(DATASET_DIR, f"{split}_data.npz"), allow_pickle=True)
    images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npz"), allow_pickle=True)

    # the two files are only row-aligned by construction (preprocessing.py writes clip_ids
    # to both)
    assert np.array_equal(data["clip_ids"], images["clip_ids"]), (
        f"{split}: clip_ids mismatch between _data.npz and _images.npz"
    )

    X, y, classes = data["X"], data["y"], data["classes"]
    crops = {crop_type: images[crop_type] for crop_type in CROP_TYPES}

    return X, y, classes, crops


def create_dataset(split: str, batch_size: int = 8, shuffle: bool = False):
    """Build a tf.data.Dataset for a split, decoding each clip's crop JPEGs lazily
    (tf.io.decode_jpeg, inside the pipeline)

    Returns:
        dataset: yields ((X, crops), y) batches, X is (batch, 32, N_LANDMARKS, HEATMAP_RESOLUTION,
            HEATMAP_RESOLUTION) float32, crops is a
            dict "left_hand"/"right_hand"/"mouth" -> (batch, 32, CROP_SIZE, CROP_SIZE, 3)
            uint8 RGB, y is (batch,) int64 label ids.
        classes: label names, (num_classes,) str, classes[y[i]] == label of sample i
    """
    X, y, classes, crops = load_split(split)
    n_frames = X.shape[1]

    def gen():
        for i in range(len(y)):
            yield X[i], {crop_type: crops[crop_type][i] for crop_type in CROP_TYPES}, y[i]

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32),
            {crop_type: tf.TensorSpec(shape=(n_frames,), dtype=tf.string) for crop_type in CROP_TYPES},
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )

    def decode(x, jpeg_crops, label):
        decoded = {
            crop_type: tf.map_fn(
                lambda b: tf.io.decode_jpeg(b, channels=3),
                jpeg_crops[crop_type],
                fn_output_signature=tf.uint8,
            )
            for crop_type in CROP_TYPES
        }
        return (x, decoded), label

    if shuffle:
        dataset = dataset.shuffle(len(y))

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, classes

def build_dataset(split: str, batch_size: int = BATCH_SIZE, shuffle: bool = False):
    """WordClassifier's actual input for a split: pose heatmaps (load_split) +
    cached hand/face DINOv3 features (load_hand_features/load_face_features) -> y.
    Everything here is already a small float array (unlike the raw JPEG crops
    create_dataset() decodes), so from_tensor_slices is enough.
    """
    X, y, classes, _crops = load_split(split)
    _, left_hand, right_hand = load_hand_features(split)
    _, mouth = load_face_features(split)

    dataset = tf.data.Dataset.from_tensor_slices(((X, left_hand, right_hand, mouth), y))
    if shuffle:
        dataset = dataset.shuffle(len(y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, classes


def train_main(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, learning_rate: float = LEARNING_RATE):
    """Compile and fit WordClassifier on train/val, saving the best checkpoint
    (by val_accuracy) to checkpoints/word_classifier_best.keras.
    """
    train_ds, classes = build_dataset("train", batch_size, shuffle=True)
    val_ds, _ = build_dataset("val", batch_size)

    model = WordClassifier(num_classes=len(classes))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, "word_classifier_best.keras"),
            monitor="val_accuracy", save_best_only=True,
        ),
    ]

    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)


if __name__ == "__main__":
    train_main()

