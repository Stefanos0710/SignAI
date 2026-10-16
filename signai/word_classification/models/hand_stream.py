"""Hand stream: frozen DINOv3 encoder (precomputed once, cached) -> trainable
temporal transformer head -> 768-d hand-stream feature.

    left/right crops (32, 112, 112, 3) uint8, per clip (preprocessing.CROP_SIZE)
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
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.models.gpu_env import ensure_gpu_env  # noqa: E402
ensure_gpu_env()  # must run before `import keras` -- see gpu_env.py. Safe even for
# --extract-features (torch, not keras): the cuDNN 8 dir it adds to LD_LIBRARY_PATH only
# contains .so.8 files, so it can't shadow torch's own cuDNN 9 (.so.9) lookup.

import numpy as np  # noqa: E402
import keras  # noqa: E402

from signai.word_classification.preprocessing import decode_image_sequence  # noqa: E402
from signai.word_classification.paths import PROCESSED_DIR  # noqa: E402
from signai.word_classification.models.progress import Progress  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

DATASET_DIR = str(PROCESSED_DIR)
CHECKPOINT_EVERY = 2000  # clips between resumable checkpoints during precompute_split

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

    def precompute_split(self, split: str):
        """Run this encoder once over every clip in `split` and cache the result
        to dataset/processed/{split}_hand_features.npz -- since the encoder's
        weights never change, its output doesn't need recomputing every
        training run (or every epoch).

        Resumable: checkpoints progress every CHECKPOINT_EVERY clips to a .partial.npz, so a
        kill/crash loses at most that many clips' GPU time, not the whole split. Skips entirely
        if the split's output already exists and matches the current clip_ids.

        Writes clip_ids (N,) int64, left_hand (N, 8, 384) float32, right_hand
        (N, 8, 384) float32 -- clip_ids matches {split}_images.npz's, same
        alignment convention as train.py::load_split.
        """
        images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npz"), allow_pickle=True)
        clip_ids = images["clip_ids"]
        n = len(clip_ids)
        out_path = os.path.join(DATASET_DIR, f"{split}_hand_features.npz")
        checkpoint_path = os.path.join(DATASET_DIR, f"{split}_hand_features.partial.npz")

        if os.path.exists(out_path):
            with np.load(out_path) as f:
                if np.array_equal(f["clip_ids"], clip_ids):
                    logging.info(f"{split}: {out_path} already up to date, skipping")
                    return

        start = 0
        left_hand = np.empty((n, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM), dtype=np.float32)
        right_hand = np.empty((n, NUM_SAMPLED_FRAMES, HAND_FEATURE_DIM), dtype=np.float32)
        if os.path.exists(checkpoint_path):
            with np.load(checkpoint_path) as f:
                if np.array_equal(f["clip_ids"], clip_ids):
                    start = int(f["next_index"])
                    left_hand[:start] = f["left_hand"][:start]
                    right_hand[:start] = f["right_hand"][:start]
                    logging.info(f"{split}: resuming hand features from clip {start:,}/{n:,}")

        # loading these once up front is required, not just faster: images[key] re-decompresses
        # and re-unpickles the *entire* array from the npz archive on every access, so indexing
        # it inside the per-clip loop would redo that (multi-GB) decompression every iteration
        left_crops = images["left_hand"]
        right_crops = images["right_hand"]

        progress = Progress(Path(checkpoint_path).with_suffix("").with_suffix(".progress.json"),
                             total=n, done=start, unit="clip")
        for i in range(start, n):
            left_hand[i] = self.extract(decode_image_sequence(left_crops[i]))
            right_hand[i] = self.extract(decode_image_sequence(right_crops[i]))
            progress.update(1)
            if (i + 1) % CHECKPOINT_EVERY == 0 or i + 1 == n:
                np.savez(checkpoint_path, clip_ids=clip_ids, left_hand=left_hand, right_hand=right_hand,
                         next_index=i + 1)
                progress.write(phase=f"hand features: {split}")
            elif progress.due():
                progress.write(phase=f"hand features: {split}", log=False)

        np.savez(out_path, clip_ids=clip_ids, left_hand=left_hand, right_hand=right_hand)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        logging.info(f"wrote {out_path}")

# --- trainable temporal head (Keras) ---------------------------------------------------------

@keras.saving.register_keras_serializable(package="signai")
class TransformerBlock(keras.layers.Layer):
    """One pre-LN transformer block (self-attention + FFN, both with residual connections).
    Shared by HandTemporalHead and FaceTemporalHead.

    Each block is its own Layer instance (not a plain dict of layers) specifically so a list of
    them is a list of tracked Layer objects -- Keras's .keras v3 save format silently fails to
    restore weights nested inside a list-of-dicts-of-layers (reloads with fresh random weights,
    no error raised), confirmed empirically; a list of real Layer subclasses round-trips correctly.
    """

    def __init__(self, dim: int, num_heads: int = 6, ffn_dim: int = 1536, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.norm1 = keras.layers.LayerNormalization()
        self.attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.norm2 = keras.layers.LayerNormalization()
        self.ffn_up = keras.layers.Dense(ffn_dim, activation="gelu")
        self.ffn_down = keras.layers.Dense(dim)

    def call(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y)
        y = self.norm2(x)
        return x + self.ffn_down(self.ffn_up(y))

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "num_heads": self.num_heads, "ffn_dim": self.ffn_dim})
        return config


@keras.saving.register_keras_serializable(package="signai")
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

    # 5) get_config round-trip: train.py's WordClassifier embeds a HandStream, and resuming
    # training depends on keras.models.load_model reconstructing it correctly -- TransformerBlock
    # being a real Layer subclass (not a plain dict of layers) is what makes this work, see its
    # docstring
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = f"{tmp_dir}/stream.keras"
        stream.save(path)
        reloaded = keras.models.load_model(path)
        reloaded_fused = reloaded([left_dummy, right_dummy])
        assert np.allclose(keras.ops.convert_to_numpy(fused), keras.ops.convert_to_numpy(reloaded_fused)), (
            "reloaded HandStream must produce identical output"
        )

    print("self-check passed (5/5 checks)")


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
