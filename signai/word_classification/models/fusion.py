"""Fusion model: the three streams' features, concatenated -> classification head.

    Pose Feature (batch, 384)   [PoseStream,  keypoints_stream.py]
    Hand Feature (batch, 768)   [HandStream,  hand_stream.py]
    Face Feature (batch, 384)   [FaceStream,  face_stream.py]
    -> concat                   (batch, 1536)
    -> Dense(512, relu) -> Dropout -> Dense(num_classes)
    -> class logits (batch, num_classes)

Inputs, one entry per stream (see each stream's module for how to produce it):
    keypoints:          (batch, T, J, 3) float32, normalized/hand-anchored (preprocessing.py)
    left_hand/right_hand: (batch, 8, 384) float32, cached DINOv3 features (hand_stream.py)
    mouth:              (batch, 8, 384) float32, cached DINOv3 features (face_stream.py)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.models.hand_stream import (  # noqa: E402
    HandStream, HAND_FEATURE_DIM, NUM_SAMPLED_FRAMES,
)
from signai.word_classification.models.face_stream import FaceStream, FACE_FEATURE_DIM  # noqa: E402
from signai.word_classification.models.keypoints_stream import PoseStream, PROJECTION_DIM  # noqa: E402

FUSED_DIM = 2 * HAND_FEATURE_DIM + FACE_FEATURE_DIM + PROJECTION_DIM  # 768 + 384 + 384 = 1536


class FusionModel(keras.Model):
    """End-to-end fusion: pose keypoints + cached hand/face DINOv3 features -> class logits.

        model = FusionModel(num_classes=100, num_landmarks=49, num_frames=32)
        logits = model([keypoints, left_hand, right_hand, mouth])

    Concatenates the three streams' features (pose 384 + hands 768 + face 384 = 1536)
    and applies one Dense(512, relu) + Dropout + Dense(num_classes) classification head.
    """

    def __init__(self, num_classes: int, num_landmarks: int = 49, num_frames: int = 32,
                 hidden_dim: int = 512, dropout: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.pose_stream = PoseStream(num_landmarks=num_landmarks, num_frames=num_frames)
        self.hand_stream = HandStream()
        self.face_stream = FaceStream()

        self.hidden = keras.layers.Dense(hidden_dim, activation="relu", name="fusion_hidden")
        self.dropout = keras.layers.Dropout(dropout, name="fusion_dropout")
        self.classifier = keras.layers.Dense(num_classes, name="fusion_classifier")

    def call(self, inputs, training=False):
        keypoints, left_hand, right_hand, mouth = inputs

        pose_feature = self.pose_stream(keypoints)                 # (batch, 384)
        hand_feature = self.hand_stream([left_hand, right_hand])   # (batch, 768)
        face_feature = self.face_stream(mouth)                     # (batch, 384)

        fused = keras.ops.concatenate([pose_feature, hand_feature, face_feature], axis=-1)  # (batch, 1536)
        x = self.hidden(fused)
        x = self.dropout(x, training=training)
        return self.classifier(x)  # (batch, num_classes)


def demo():
    """Self-check: python signai/word_classification/models/fusion.py --self-check
    Fast -- no DINOv3 download or on-disk dataset needed, FusionModel is exercised with
    random dummy features standing in for the three streams' real inputs.
    """
    rng = np.random.default_rng(0)
    batch, num_frames, num_landmarks, num_classes = 4, 32, 49, 10

    keypoints = rng.normal(size=(batch, num_frames, num_landmarks, 3)).astype(np.float32)
    left_hand = rng.normal(size=(batch, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM)).astype(np.float32)
    right_hand = rng.normal(size=(batch, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM)).astype(np.float32)
    mouth = rng.normal(size=(batch, NUM_SAMPLED_FRAMES, FACE_FEATURE_DIM)).astype(np.float32)

    model = FusionModel(num_classes=num_classes, num_landmarks=num_landmarks, num_frames=num_frames)
    logits = model([keypoints, left_hand, right_hand, mouth])
    assert logits.shape == (batch, num_classes)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(logits)))

    print("self-check passed (1/1 checks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    args = parser.parse_args()

    if args.self_check:
        demo()
