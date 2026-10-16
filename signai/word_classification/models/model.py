"""Top-level word classifier: pose keypoints + cached hand/face DINOv3 features -> class logits.

Reuses fusion.py's fusion head (concat -> Dense -> Dropout -> Dense) but takes the pose
branch's input in the shape it actually has on disk (dataset/processed/*_data.npz: X is
(N, 32, 147) flattened keypoints) and renders the per-landmark heatmaps on the fly with
heatmap.py's HeatmapGenerator, like keypoints_stream.py's PoseStream -- baked into the dataset
they'd be 58 MB per clip.

    model = WordClassifier(num_classes=100)
    logits = model([keypoints, left_hand, right_hand, mouth])

Inputs:
    keypoints:           (batch, T, J * 3) float32, from *_data.npz's X (preprocessing.py)
    left_hand/right_hand: (batch, 8, 384) float32, cached DINOv3 features (hand_stream.py)
    mouth:               (batch, 8, 384) float32, cached DINOv3 features (face_stream.py)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.models.gpu_env import ensure_gpu_env  # noqa: E402
ensure_gpu_env()  # must run before `import keras` (which imports tensorflow) -- see gpu_env.py

import numpy as np  # noqa: E402
import keras  # noqa: E402

from signai.word_classification.models.heatmap import (  # noqa: E402
    HeatmapGenerator, HEATMAP_HEIGHT, HEATMAP_WIDTH, HEATMAP_SIGMA,
)
from signai.word_classification.models.keypoints_stream import (  # noqa: E402
    build_pose_encoder, PoseFeatureProjection, PROJECTION_DIM, reshape_keypoints,
)
from signai.word_classification.models.hand_stream import (  # noqa: E402
    HandStream, HAND_FEATURE_DIM, NUM_SAMPLED_FRAMES,
)
from signai.word_classification.models.face_stream import FaceStream, FACE_FEATURE_DIM  # noqa: E402

FUSED_DIM = PROJECTION_DIM + 2 * HAND_FEATURE_DIM + FACE_FEATURE_DIM  # 384 + 768 + 384 = 1536


@keras.saving.register_keras_serializable(package="signai")
class WordClassifier(keras.Model):
    """End-to-end word classifier: pose keypoints + cached hand/face DINOv3 features ->
    class logits. See module docstring for the exact shapes each stream expects.
    """

    def __init__(self, num_classes: int, num_landmarks: int = 49, num_frames: int = 32,
                 height: int = HEATMAP_HEIGHT, width: int = HEATMAP_WIDTH, sigma: float = HEATMAP_SIGMA,
                 hidden_dim: int = 512, dropout: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        # stashed verbatim for get_config -- a subclassed Model isn't reconstructible from a saved
        # .keras file otherwise (keras.models.load_model needs these to call cls(**config)), which
        # train.py's resume-from-checkpoint relies on
        self.num_classes = num_classes
        self.num_landmarks = num_landmarks
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.sigma = sigma
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.heatmap_generator = HeatmapGenerator(height, width, sigma, channels_last=True)
        self.pose_encoder = build_pose_encoder(num_landmarks, height, width, num_frames)
        self.pose_projection = PoseFeatureProjection(PROJECTION_DIM)
        self.hand_stream = HandStream()
        self.face_stream = FaceStream()

        self.hidden = keras.layers.Dense(hidden_dim, activation="relu", name="fusion_hidden")
        self.dropout = keras.layers.Dropout(dropout, name="fusion_dropout")
        self.classifier = keras.layers.Dense(num_classes, name="fusion_classifier")

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "num_landmarks": self.num_landmarks,
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "sigma": self.sigma,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
        })
        return config

    def call(self, inputs, training=False):
        keypoints, left_hand, right_hand, mouth = inputs

        keypoints = reshape_keypoints(keypoints, self.num_landmarks)  # (batch,T,J*3) -> (batch,T,J,3)
        heatmaps = self.heatmap_generator(keypoints)                    # -> (batch,T,H,W,J)
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

    keypoints = rng.normal(size=(batch, num_frames, num_landmarks * 3)).astype(np.float32)
    left_hand = rng.normal(size=(batch, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM)).astype(np.float32)
    right_hand = rng.normal(size=(batch, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM)).astype(np.float32)
    mouth = rng.normal(size=(batch, NUM_SAMPLED_FRAMES, FACE_FEATURE_DIM)).astype(np.float32)

    model = WordClassifier(num_classes=num_classes, num_landmarks=num_landmarks, num_frames=num_frames)
    logits = model([keypoints, left_hand, right_hand, mouth])
    assert logits.shape == (batch, num_classes)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(logits)))

    # get_config round-trip: train.py's resume-from-checkpoint depends on keras.models.load_model
    # being able to reconstruct WordClassifier (a subclassed Model) from a saved .keras file --
    # without get_config this raises "missing 1 required positional argument: 'num_classes'"
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = f"{tmp_dir}/model.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        reloaded_logits = reloaded([keypoints, left_hand, right_hand, mouth])
        assert reloaded_logits.shape == (batch, num_classes)
        assert np.allclose(keras.ops.convert_to_numpy(logits), keras.ops.convert_to_numpy(reloaded_logits)), (
            "reloaded model must produce identical output"
        )

    print("self-check passed (2/2 checks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    args = parser.parse_args()

    if args.self_check:
        demo()
