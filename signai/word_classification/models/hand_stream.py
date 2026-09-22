"""Hand stream: frozen DINOv3 encoder (precomputed once, cached) -> trainable
temporal transformer head -> 768-d hand-stream feature.

    left/right crops (32, 224, 224, 3) uint8, per clip
    -> sample 8 frames (sample_frame_indices)
    -> DINOv3 ViT-S/16, frozen               (DinoHandEncoder)
    -> cached (8, 384) float32 per hand      (dataset/processed/{split}_hand_features.npz)
    -> HandTemporalHead, shared weights      (trainable)
    -> concat(left 384-d, right 384-d)       (HandStream)
    -> Hand Feature (batch, 768)

Crops are loaded from signai/word_classification/dataset/processed/*_images.npz
(written by preprocessing.py -- see CROP_TYPES/decode_image_sequence there).
DINOv3 weights are gated on Hugging Face: accept the license on
https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m and run
`huggingface-cli login` (or set HF_TOKEN) before --extract-features works.
Only DinoHandEncoder touches torch/transformers -- the trainable Keras
pieces (HandTemporalHead, HandStream) never need torch installed, they only
consume the cached float32 features.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.preprocessing import decode_image_sequence  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed")

HAND_FEATURE_DIM = 384
NUM_SAMPLED_FRAMES = 8
DINOV3_CHECKPOINT = "facebook/dinov3-vits16-pretrain-lvd1689m"


def load_hand_crops(split: str = "test", clip_index: int | None = None):
    """Load one clip's decoded left/right hand crop sequences from
    dataset/processed/{split}_images.npz.

    clip_index: row into that split; a random one is picked if not given.

    Returns (left_hand, right_hand), each (MAX_FRAMES, CROP_SIZE, CROP_SIZE, 3)
    uint8 RGB (decode_image_sequence's output).
    """
    images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npz"), allow_pickle=True)
    if clip_index is None:
        clip_index = np.random.randint(len(images["clip_ids"]))
    left_hand = decode_image_sequence(images["left_hand"][clip_index])
    right_hand = decode_image_sequence(images["right_hand"][clip_index])
    return left_hand, right_hand


def sample_frame_indices(num_frames: int = 32, num_sampled: int = NUM_SAMPLED_FRAMES):
    """`num_sampled` evenly-spaced frame indices over [0, num_frames), including
    both ends. Shared by DinoHandEncoder.precompute_split and anything else that
    needs to know exactly which source frames the cached DINOv3 features came from.
    """
    return np.linspace(0, num_frames - 1, num_sampled).round().astype(int)


def load_hand_features(split: str = "test"):
    """Load the cached DINOv3 features written by DinoHandEncoder.precompute_split,
    verified row-aligned with {split}_images.npz (same clip_ids check
    train.py::load_split does for the crops/keypoints npz pair).

    Returns clip_ids (N,), left_hand (N, 8, 384) float32, right_hand (N, 8, 384) float32.
    """
    features = np.load(os.path.join(DATASET_DIR, f"{split}_hand_features.npz"))
    images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npz"), allow_pickle=True)
    assert np.array_equal(features["clip_ids"], images["clip_ids"]), (
        f"{split}: clip_ids mismatch between _hand_features.npz and _images.npz -- "
        f"rerun --extract-features {split}"
    )
    return features["clip_ids"], features["left_hand"], features["right_hand"]


# frozen DINOv3 encoder

class DinoHandEncoder:
    """Loads the frozen DINOv3 processor/model once (on first use) and runs it
    over hand crop sequences. Only this class touches torch/transformers --
    importing hand_stream.py for the trainable Keras pieces (HandTemporalHead,
    HandStream) never requires torch to be installed, only instantiating this
    class does.

        encoder = DinoHandEncoder()
        features = encoder.extract(crops)          # one hand, one clip -> (8, 384)
        encoder.precompute_split("test")            # every clip in a split -> cached npz
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
        """One hand's one clip: (32, CROP_SIZE, CROP_SIZE, 3) uint8 RGB -> sample 8
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
        to dataset/processed/{split}_hand_features.npz -- since the encoder's
        weights never change, its output doesn't need recomputing every
        training run (or every epoch).

        Writes clip_ids (N,) int64, left_hand (N, 8, 384) float32, right_hand
        (N, 8, 384) float32 -- clip_ids matches {split}_images.npz's, same
        alignment convention as train.py::load_split.
        """
        images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npz"), allow_pickle=True)
        clip_ids = images["clip_ids"]
        n = len(clip_ids)

        left_hand = np.empty((n, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM), dtype=np.float32)
        right_hand = np.empty((n, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM), dtype=np.float32)

        for i in range(n):
            left_hand[i] = self.extract(decode_image_sequence(images["left_hand"][i]))
            right_hand[i] = self.extract(decode_image_sequence(images["right_hand"][i]))
            if (i + 1) % log_every == 0 or i + 1 == n:
                print(f"{split}: {i + 1}/{n} clips", end="\r")
        print()

        out_path = os.path.join(DATASET_DIR, f"{split}_hand_features.npz")
        np.savez(out_path, clip_ids=clip_ids, left_hand=left_hand, right_hand=right_hand)
        print(f"wrote {out_path}")

# --- trainable temporal head (Keras) ---------------------------------------------------------

class HandTemporalHead(keras.layers.Layer):
    """Trainable pre-LN transformer over one hand's (batch, 8, 384) DINOv3
    features: prepend a learned CLS token, add learned positional embeddings,
    2 transformer blocks (6 heads, FFN 1536), return the CLS token's output.

    Meant to be called on both hands with the *same* layer instance (weight
    sharing) -- see HandStream.
    """

    def __init__(self, num_frames: int = NUM_SAMPLED_FRAMES, dim: int = HAND_FEATURE_DIM,
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


class HandStream(keras.Model):
    """End-to-end hand stream: cached per-hand DINOv3 features -> 768-d hand
    feature.

        hand_stream = HandStream()
        hand_features = hand_stream([left, right])  # 2x (batch, 8, 384) -> (batch, 768)

    Applies one shared HandTemporalHead to both hands (weight sharing), then
    concatenates the two 384-d outputs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temporal_head = HandTemporalHead()

    def call(self, inputs):
        left, right = inputs
        left_feature = self.temporal_head(left)    # (batch, 384)
        right_feature = self.temporal_head(right)  # (batch, 384), same weights
        return keras.ops.concatenate([left_feature, right_feature], axis=-1)  # (batch, 768)


def demo():
    """Self-check: python signai/word_classification/models/hand_stream.py --self-check
    Fast -- no DINOv3 download needed, HandTemporalHead/HandStream are exercised
    with random dummy features. Use --extract-features to run the real encoder.
    """
    # 1) load_hand_crops: real on-disk crops
    left_hand, right_hand = load_hand_crops("test", clip_index=0)
    assert left_hand.shape == right_hand.shape
    assert left_hand.ndim == 4 and left_hand.shape[-1] == 3
    assert left_hand.dtype == np.uint8

    # 2) sample_frame_indices: 8 evenly-spaced indices, endpoints included
    idx = sample_frame_indices(32, 8)
    assert idx.shape == (8,) and idx[0] == 0 and idx[-1] == 31

    # 3) HandTemporalHead: (batch, 8, 384) -> (batch, 384)
    rng = np.random.default_rng(0)
    dummy = rng.normal(size=(4, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM)).astype(np.float32)
    head = HandTemporalHead()
    out = head(dummy)
    assert out.shape == (4, HAND_FEATURE_DIM)

    # 4) HandStream: two (batch, 8, 384) hands -> (batch, 768), shared weights
    stream = HandStream()
    left_dummy = rng.normal(size=(4, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM)).astype(np.float32)
    right_dummy = rng.normal(size=(4, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM)).astype(np.float32)
    fused = stream([left_dummy, right_dummy])
    assert fused.shape == (4, 2 * HAND_FEATURE_DIM)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(fused)))

    print("self-check passed (4/4 checks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    parser.add_argument("--extract-features", metavar="SPLIT",
                         help="run the frozen DINOv3 encoder over a split (train/val/test) and "
                              "cache the result to dataset/processed/{split}_hand_features.npz")
    args = parser.parse_args()

    if args.self_check:
        demo()

    if args.extract_features:
        DinoHandEncoder().precompute_split(args.extract_features)
