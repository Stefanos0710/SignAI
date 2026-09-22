"""Top-level word classifier: pose heatmaps + cached hand/face DINOv3 features -> class logits.

Reuses fusion.py's fusion head (concat -> Dense -> Dropout -> Dense) but takes the pose
branch's input in the shape it actually has on disk. preprocessing.py renders heatmaps
once at preprocessing time (dataset/processed/*_data.npz: X is (N, 32, 49, 96, 96),
channels-first), unlike keypoints_stream.py's PoseStream, which expects raw (T, J, 3)
keypoints and renders heatmaps itself.

    model = WordClassifier(num_classes=100)
    logits = model([heatmaps, left_hand, right_hand, mouth])

Inputs:
    heatmaps:            (batch, T, J, H, W) float32, from *_data.npz's X (preprocessing.py)
    left_hand/right_hand: (batch, 8, 384) float32, cached DINOv3 features (hand_stream.py)
    mouth:               (batch, 8, 384) float32, cached DINOv3 features (face_stream.py)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.models.keypoints_stream import (  # noqa: E402
    build_pose_encoder, PoseFeatureProjection, PROJECTION_DIM,
    HEATMAP_HEIGHT, HEATMAP_WIDTH,
)
from signai.word_classification.models.hand_stream import (  # noqa: E402
    HandStream, HAND_FEATURE_DIM, NUM_SAMPLED_FRAMES,
)
from signai.word_classification.models.face_stream import FaceStream, FACE_FEATURE_DIM  # noqa: E402

FUSED_DIM = PROJECTION_DIM + 2 * HAND_FEATURE_DIM + FACE_FEATURE_DIM  # 384 + 768 + 384 = 1536


class WordClassifier(keras.Model):
    """End-to-end word classifier: pose heatmaps + cached hand/face DINOv3 features ->
    class logits. See module docstring for the exact shapes each stream expects.
    """

    def __init__(self, num_classes: int, num_landmarks: int = 49, num_frames: int = 32,
                 height: int = HEATMAP_HEIGHT, width: int = HEATMAP_WIDTH,
                 hidden_dim: int = 512, dropout: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.pose_encoder = build_pose_encoder(num_landmarks, height, width, num_frames)
        self.pose_projection = PoseFeatureProjection(PROJECTION_DIM)
        self.hand_stream = HandStream()
        self.face_stream = FaceStream()

        self.hidden = keras.layers.Dense(hidden_dim, activation="relu", name="fusion_hidden")
        self.dropout = keras.layers.Dropout(dropout, name="fusion_dropout")
        self.classifier = keras.layers.Dense(num_classes, name="fusion_classifier")

    def call(self, inputs, training=False):
        heatmaps, left_hand, right_hand, mouth = inputs

        heatmaps = keras.ops.transpose(heatmaps, (0, 1, 3, 4, 2))  # (batch,T,J,H,W) -> (batch,T,H,W,J)
        pose_feature = self.pose_projection(self.pose_encoder(heatmaps))  # (batch, 384)
        hand_feature = self.hand_stream([left_hand, right_hand])          # (batch, 768)
        face_feature = self.face_stream(mouth)                            # (batch, 384)

        fused = keras.ops.concatenate([pose_feature, hand_feature, face_feature], axis=-1)  # (batch, 1536)
        x = self.hidden(fused)
        x = self.dropout(x, training=training)
        return self.classifier(x)  # (batch, num_classes)


def demo():
    """Self-check: python signai/word_classification/models/model.py --self-check
    """
    rng = np.random.default_rng(0)
    batch, num_frames, num_landmarks, num_classes = 4, 32, 49, 10

    heatmaps = rng.uniform(size=(batch, num_frames, num_landmarks, HEATMAP_HEIGHT, HEATMAP_WIDTH)).astype(np.float32)
    left_hand = rng.normal(size=(batch, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM)).astype(np.float32)
    right_hand = rng.normal(size=(batch, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM)).astype(np.float32)
    mouth = rng.normal(size=(batch, NUM_SAMPLED_FRAMES, FACE_FEATURE_DIM)).astype(np.float32)

    model = WordClassifier(num_classes=num_classes, num_landmarks=num_landmarks, num_frames=num_frames)
    logits = model([heatmaps, left_hand, right_hand, mouth])
    assert logits.shape == (batch, num_classes)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(logits)))

    print("self-check passed (1/1 checks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    args = parser.parse_args()

    if args.self_check:
        demo()
