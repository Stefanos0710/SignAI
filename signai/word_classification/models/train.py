"""
1) Loading the dataset
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.preprocessing import decode_image_sequence  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed")

SPLITS = ["train", "val", "test"]
CROP_TYPES = ("left_hand", "right_hand", "mouth")


def load_split(split: str):
    """Load a dataset split (train/val/test) from dataset/processed/.

    Returns:
        X: keypoints, (N, 32, 147) float32
        y: label ids, (N,) int64
        classes: label names, (num_classes,) str, classes[y[i]] == label of sample i
        crops: dict with "left_hand"/"right_hand"/"mouth", each (N, 32, CROP_SIZE, CROP_SIZE, 3)
            uint8 RGB, decoded from the JPEG bytes stored in {split}_images.npz
    """
    data = np.load(os.path.join(DATASET_DIR, f"{split}_data.npz"), allow_pickle=True)
    images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npz"), allow_pickle=True)

    # the two files are only row-aligned by construction (preprocessing.py writes clip_ids
    # to both) -- verify that instead of assuming it.
    assert np.array_equal(data["clip_ids"], images["clip_ids"]), (
        f"{split}: clip_ids mismatch between _data.npz and _images.npz"
    )

    X, y, classes = data["X"], data["y"], data["classes"]
    crops = {
        crop_type: np.stack([decode_image_sequence(seq) for seq in images[crop_type]])
        for crop_type in CROP_TYPES
    }

    return X, y, classes, crops

def main():
    pass

if __name__ == "__main__":
    load_split("train")