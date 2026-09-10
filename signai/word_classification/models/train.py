"""
1) Loading the dataset:
       - load_split(split: str)      -- read the npz files, keep crop JPEGs undecoded
       - create_dataset(split: str)  -- wrap load_split into a tf.data.Dataset that decodes
                                         each clip's crops lazily, per batch


"""

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed")

SPLITS = ["train", "val", "test"]
CROP_TYPES = ("left_hand", "right_hand", "mouth")

def generate_heatmap():
    pass

def load_split(split: str):
    """Load a dataset split (train/val/test) from dataset/processed/.

    Crops are returned as the raw JPEG byte strings stored in {split}_images.npz, *not*
    decoded to pixels here. Decoding every clip directly is what is flooding RAM.
    create_dataset() decodes lazily, one batch at a time, via tf.io.decode_jpeg.

    Returns:
        X: keypoints, (N, 32, 147) float32
        y: label ids, (N,) int64
        classes: label names, (num_classes,) str, classes[y[i]] == label of sample i
        crops: dict with "left_hand"/"right_hand"/"mouth", each (N, 32) object array of
            JPEG-encoded bytes
    """
    data = np.load(os.path.join(DATASET_DIR, f"{split}_data.npz"), allow_pickle=True)
    images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npz"), allow_pickle=True)

    # the two files are only row-aligned by construction (preprocessing.py writes clip_ids
    # to both) -- verify that instead of assuming it.
    assert np.array_equal(data["clip_ids"], images["clip_ids"]), (
        f"{split}: clip_ids mismatch between _data.npz and _images.npz"
    )

    X, y, classes = data["X"], data["y"], data["classes"]
    crops = {crop_type: images[crop_type] for crop_type in CROP_TYPES}

    return X, y, classes, crops


def create_dataset(split: str, batch_size: int = 8, shuffle: bool = False):
    """Build a tf.data.Dataset for a split, decoding each clip's crop JPEGs lazily
    (tf.io.decode_jpeg, inside the pipeline) instead of decoding the whole split into RAM
    up front -- only the samples currently in flight are ever materialized as pixels.

    Returns:
        dataset: yields ((X, crops), y) batches, X is (batch, 32, 147) float32, crops is a
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


def main():
    pass


if __name__ == "__main__":
    ds, classes = create_dataset("train")
    print(f"train dataset: {ds}, classes: {classes}")
