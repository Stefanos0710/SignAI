"""Face stream: frozen DINOv3 encoder (precomputed once, cached) -> trainable
temporal transformer head -> 384-d face-stream feature.

    mouth crops (32, 224, 224, 3) uint8, per clip
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
import os
import sys
from pathlib import Path

import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.preprocessing import decode_image_sequence  # noqa: E402
from signai.word_classification.models.hand_stream import (  # noqa: E402
    DINOV3_CHECKPOINT,
    NUM_SAMPLED_FRAMES,
    sample_frame_indices,
)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed")

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

    def precompute_split(self, split: str, log_every: int = 16):
        """Run this encoder once over every clip in `split` and cache the result
        to dataset/processed/{split}_face_features.npz -- since the encoder's
        weights never change, its output doesn't need recomputing every
        training run (or every epoch).

        Writes clip_ids (N,) int64, mouth (N, 8, 384) float32 -- clip_ids matches
        {split}_images.npz's, same alignment convention as train.py::load_split.
        """
        images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npz"), allow_pickle=True)
        clip_ids = images["clip_ids"]
        n = len(clip_ids)

        mouth = np.empty((n, NUM_SAMPLED_FRAMES, FACE_FEATURE_DIM), dtype=np.float32)

        for i in range(n):
            mouth[i] = self.extract(decode_image_sequence(images["mouth"][i]))
            if (i + 1) % log_every == 0 or i + 1 == n:
                print(f"{split}: {i + 1}/{n} clips", end="\r")
        print()

        out_path = os.path.join(DATASET_DIR, f"{split}_face_features.npz")
        np.savez(out_path, clip_ids=clip_ids, mouth=mouth)
        print(f"wrote {out_path}")

# --- trainable temporal head (Keras) ---------------------------------------------------------

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

        self.cls_token = self.add_weight(
            name="cls_token", shape=(1, 1, dim), initializer="random_normal", trainable=True,
        )
        self.pos_embedding = self.add_weight(
            name="pos_embedding", shape=(1, num_frames + 1, dim), initializer="random_normal",
            trainable=True,
        )

        self.blocks = []
        for i in range(num_layers):
            self.blocks.append(dict(
                norm1=keras.layers.LayerNormalization(name=f"block{i}_norm1"),
                attn=keras.layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=dim // num_heads, name=f"block{i}_attn"
                ),
                norm2=keras.layers.LayerNormalization(name=f"block{i}_norm2"),
                ffn_up=keras.layers.Dense(ffn_dim, activation="gelu", name=f"block{i}_ffn_up"),
                ffn_down=keras.layers.Dense(dim, name=f"block{i}_ffn_down"),
            ))
        self.final_norm = keras.layers.LayerNormalization(name="final_norm")

    def call(self, x):
        # (batch, 8, 384) -> (batch, 9, 384), CLS prepended
        batch_size = keras.ops.shape(x)[0]
        cls = keras.ops.repeat(self.cls_token, batch_size, axis=0)
        x = keras.ops.concatenate([cls, x], axis=1)
        x = x + self.pos_embedding

        for block in self.blocks:
            y = block["norm1"](x)
            x = x + block["attn"](y, y)
            y = block["norm2"](x)
            x = x + block["ffn_down"](block["ffn_up"](y))

        x = self.final_norm(x)
        return x[:, 0, :]  # CLS token -> (batch, 384)


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

    print("self-check passed (3/3 checks)")


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
