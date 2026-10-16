"""Face stream: frozen DINOv3 encoder (precomputed once, cached) -> trainable
temporal transformer head -> 384-d face-stream feature.

    mouth crops (32, 112, 112, 3) uint8, per clip (preprocessing.CROP_SIZE)
    -> sample 8 frames (hand_stream.sample_frame_indices)
    -> DINOv3 ViT-S/16, frozen               (DinoFaceEncoder)
    -> cached (8, 384) float32               (dataset/processed/{split}_face_features.npz)
    -> FaceTemporalHead                      (trainable)
    -> Face Feature (batch, 384)

Only one crop type ("mouth", see preprocessing.py) unlike hand_stream's paired
left/right hands, so there's no weight-sharing dual call here -- one crop in,
one feature out. Everything else (encoder, frame sampling, temporal head
architecture) mirrors hand_stream.py; see that file for the shared rationale.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.models.gpu_env import ensure_gpu_env  # noqa: E402
ensure_gpu_env()  # must run before `import keras` -- see gpu_env.py and hand_stream.py's own note

import numpy as np  # noqa: E402
import keras  # noqa: E402

from signai.word_classification.preprocessing import decode_image_sequence  # noqa: E402
from signai.word_classification.models.hand_stream import (  # noqa: E402
    CHECKPOINT_EVERY,
    DINOV3_CHECKPOINT,
    NUM_SAMPLED_FRAMES,
    TransformerBlock,
    sample_frame_indices,
)
from signai.word_classification.models.progress import Progress  # noqa: E402
from signai.word_classification.paths import PROCESSED_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

DATASET_DIR = str(PROCESSED_DIR)

FACE_FEATURE_DIM = 384


def load_face_crops(split: str = "test", clip_index: int | None = None):
    """Load one clip's decoded mouth crop sequence from dataset/processed/{split}_images.npz.

    clip_index: row into that split; a random one is picked if not given.

    Returns (MAX_FRAMES, CROP_SIZE, CROP_SIZE, 3) uint8 RGB (decode_image_sequence's output).
    """
    images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npz"), allow_pickle=True)
    if clip_index is None:
        clip_index = np.random.randint(len(images["clip_ids"]))
    return decode_image_sequence(images["mouth"][clip_index])


def load_face_features(split: str = "test"):
    """Load the cached DINOv3 features written by DinoFaceEncoder.precompute_split,
    verified row-aligned with {split}_images.npz (same clip_ids check
    train.py::load_split does for the crops/keypoints npz pair).

    Returns clip_ids (N,), mouth (N, 8, 384) float32.
    """
    features = np.load(os.path.join(DATASET_DIR, f"{split}_face_features.npz"))
    images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npz"), allow_pickle=True)
    assert np.array_equal(features["clip_ids"], images["clip_ids"]), (
        f"{split}: clip_ids mismatch between _face_features.npz and _images.npz -- "
        f"rerun --extract-features {split}"
    )
    return features["clip_ids"], features["mouth"]


# frozen DINOv3 encoder

class DinoFaceEncoder:
    """Loads the frozen DINOv3 processor/model once (on first use) and runs it
    over mouth crop sequences. Only this class touches torch/transformers --
    importing face_stream.py for the trainable Keras pieces (FaceTemporalHead,
    FaceStream) never requires torch to be installed, only instantiating this
    class does.

        encoder = DinoFaceEncoder()
        features = encoder.extract(crops)   # one clip -> (8, 384)
        encoder.precompute_split("test")     # every clip in a split -> cached npz
    """

    def __init__(self, checkpoint: str = DINOV3_CHECKPOINT):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def extract(self, crops):
        """One clip's mouth crops: (32, CROP_SIZE, CROP_SIZE, 3) uint8 RGB -> sample 8
        frames -> frozen DINOv3 forward pass -> (8, 384) float32 CLS-token features.
        """
        import torch

        frame_idx = sample_frame_indices(crops.shape[0])
        sampled = [crops[i] for i in frame_idx]

        inputs = self.processor(images=sampled, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # (8, 384), one CLS token per frame
        return cls_tokens.cpu().numpy().astype(np.float32)

    def precompute_split(self, split: str):
        """Run this encoder once over every clip in `split` and cache the result
        to dataset/processed/{split}_face_features.npz -- since the encoder's
        weights never change, its output doesn't need recomputing every
        training run (or every epoch).

        Resumable: checkpoints progress every CHECKPOINT_EVERY clips to a .partial.npz, so a
        kill/crash loses at most that many clips' GPU time, not the whole split. Skips entirely
        if the split's output already exists and matches the current clip_ids.

        Writes clip_ids (N,) int64, mouth (N, 8, 384) float32 -- clip_ids matches
        {split}_images.npz's, same alignment convention as train.py::load_split.
        """
        images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npz"), allow_pickle=True)
        clip_ids = images["clip_ids"]
        n = len(clip_ids)
        out_path = os.path.join(DATASET_DIR, f"{split}_face_features.npz")
        checkpoint_path = os.path.join(DATASET_DIR, f"{split}_face_features.partial.npz")

        if os.path.exists(out_path):
            with np.load(out_path) as f:
                if np.array_equal(f["clip_ids"], clip_ids):
                    logging.info(f"{split}: {out_path} already up to date, skipping")
                    return

        start = 0
        mouth = np.empty((n, NUM_SAMPLED_FRAMES, FACE_FEATURE_DIM), dtype=np.float32)
        if os.path.exists(checkpoint_path):
            with np.load(checkpoint_path) as f:
                if np.array_equal(f["clip_ids"], clip_ids):
                    start = int(f["next_index"])
                    mouth[:start] = f["mouth"][:start]
                    logging.info(f"{split}: resuming face features from clip {start:,}/{n:,}")

        # loading this once up front is required, not just faster: images["mouth"]
        # re-decompresses and re-unpickles the *entire* array from the npz archive on every
        # access, so indexing it inside the per-clip loop would redo that (multi-GB)
        # decompression every iteration
        mouth_crops = images["mouth"]

        progress = Progress(Path(checkpoint_path).with_suffix("").with_suffix(".progress.json"),
                             total=n, done=start, unit="clip")
        for i in range(start, n):
            mouth[i] = self.extract(decode_image_sequence(mouth_crops[i]))
            progress.update(1)
            if (i + 1) % CHECKPOINT_EVERY == 0 or i + 1 == n:
                np.savez(checkpoint_path, clip_ids=clip_ids, mouth=mouth, next_index=i + 1)
                progress.write(phase=f"face features: {split}")
            elif progress.due():
                progress.write(phase=f"face features: {split}", log=False)

        np.savez(out_path, clip_ids=clip_ids, mouth=mouth)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        logging.info(f"wrote {out_path}")

# --- trainable temporal head (Keras) ---------------------------------------------------------

@keras.saving.register_keras_serializable(package="signai")
class FaceTemporalHead(keras.layers.Layer):
    """Trainable pre-LN transformer over the mouth's (batch, 8, 384) DINOv3
    features: prepend a learned CLS token, add learned positional embeddings,
    2 transformer blocks (6 heads, FFN 1536), return the CLS token's output.
    """

    def __init__(self, num_frames: int = NUM_SAMPLED_FRAMES, dim: int = FACE_FEATURE_DIM,
                 num_heads: int = 6, ffn_dim: int = 1536, num_layers: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers

        self.cls_token = self.add_weight(
            name="cls_token", shape=(1, 1, dim), initializer="random_normal", trainable=True,
        )
        self.pos_embedding = self.add_weight(
            name="pos_embedding", shape=(1, num_frames + 1, dim), initializer="random_normal",
            trainable=True,
        )
        self.blocks = [TransformerBlock(dim, num_heads, ffn_dim, name=f"block{i}") for i in range(num_layers)]
        self.final_norm = keras.layers.LayerNormalization(name="final_norm")

    def call(self, x):
        # (batch, 8, 384) -> (batch, 9, 384), CLS prepended
        batch_size = keras.ops.shape(x)[0]
        cls = keras.ops.repeat(self.cls_token, batch_size, axis=0)
        x = keras.ops.concatenate([cls, x], axis=1)
        x = x + self.pos_embedding

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return x[:, 0, :]  # CLS token -> (batch, 384)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_frames": self.num_frames, "dim": self.dim, "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim, "num_layers": self.num_layers,
        })
        return config


@keras.saving.register_keras_serializable(package="signai")
class FaceStream(keras.Model):
    """End-to-end face stream: cached mouth DINOv3 features -> 384-d face feature.

        face_stream = FaceStream()
        face_feature = face_stream(mouth)  # (batch, 8, 384) -> (batch, 384)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temporal_head = FaceTemporalHead()

    def call(self, mouth):
        return self.temporal_head(mouth)  # (batch, 384)


def demo():
    """Self-check: python signai/word_classification/models/face_stream.py --self-check
    Fast -- no DINOv3 download needed, FaceTemporalHead/FaceStream are exercised
    with random dummy features. Use --extract-features to run the real encoder.
    """
    # 1) load_face_crops: real on-disk crops
    mouth = load_face_crops("test", clip_index=0)
    assert mouth.ndim == 4 and mouth.shape[-1] == 3
    assert mouth.dtype == np.uint8

    # 2) FaceTemporalHead: (batch, 8, 384) -> (batch, 384)
    rng = np.random.default_rng(0)
    dummy = rng.normal(size=(4, NUM_SAMPLED_FRAMES, FACE_FEATURE_DIM)).astype(np.float32)
    head = FaceTemporalHead()
    out = head(dummy)
    assert out.shape == (4, FACE_FEATURE_DIM)

    # 3) FaceStream: (batch, 8, 384) -> (batch, 384)
    stream = FaceStream()
    fused = stream(dummy)
    assert fused.shape == (4, FACE_FEATURE_DIM)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(fused)))

    # 4) get_config round-trip -- see hand_stream.py's equivalent check for why this matters
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = f"{tmp_dir}/stream.keras"
        stream.save(path)
        reloaded = keras.models.load_model(path)
        reloaded_fused = reloaded(dummy)
        assert np.allclose(keras.ops.convert_to_numpy(fused), keras.ops.convert_to_numpy(reloaded_fused)), (
            "reloaded FaceStream must produce identical output"
        )

    print("self-check passed (4/4 checks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    parser.add_argument("--extract-features", metavar="SPLIT",
                         help="run the frozen DINOv3 encoder over a split (train/val/test) and "
                              "cache the result to dataset/processed/{split}_face_features.npz")
    args = parser.parse_args()

    if args.self_check:
        demo()

    if args.extract_features:
        DinoFaceEncoder().precompute_split(args.extract_features)
