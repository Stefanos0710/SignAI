import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
import keras
import jiwer


DEFAULT_HISTORY_IMAGE = Path("/loctmp/zzm01651/SignAI/models/training_history_v36.png")
DEFAULT_TOKENIZER_PATH = Path("/loctmp/zzm01651/SignAI/tokenizers/gloss_tokenizer.json")
DEFAULT_DATA_FOLDER = Path("data/train_data")
DEFAULT_MODELS_DIR = Path("models")


def encoder_mask_fn(t):
    return tf.reduce_sum(tf.abs(t), axis=-1) > 1e-6


def create_cross_mask(inputs):
    dec_mask, enc_mask = inputs
    dec_mask = tf.cast(dec_mask[:, :, tf.newaxis], tf.bool)
    enc_mask = tf.cast(enc_mask[:, tf.newaxis, :], tf.bool)
    return tf.cast(tf.logical_and(dec_mask, enc_mask), tf.bool)


def load_train_module(repo_root: Path):
    train_path = repo_root / "train-seq2seq.py"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find training module: {train_path}")

    spec = importlib.util.spec_from_file_location("train_seq2seq_module", str(train_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create import spec for train-seq2seq.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(repo_root: Path, path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def parse_checkpoint_epoch(name: str) -> int:
    # expected: checkpoint_v36_epoch_53.keras
    try:
        stem = Path(name).stem
        epoch_part = stem.split("_epoch_")[-1]
        return int(epoch_part)
    except Exception:
        return -1


def find_latest_v36_model(models_dir: Path) -> Path:
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    checkpoints = sorted(
        models_dir.glob("checkpoint_v36_epoch_*.keras"),
        key=lambda p: parse_checkpoint_epoch(p.name),
    )
    if checkpoints:
        return checkpoints[-1]

    trained = models_dir / "trained_model_v36.keras"
    if trained.exists():
        return trained

    raise FileNotFoundError(
        "No v36 model found. Expected one of: models/checkpoint_v36_epoch_*.keras or models/trained_model_v36.keras"
    )


def load_tokenizer(tokenizer_path: Path):
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    if isinstance(tokenizer_json, dict):
        tokenizer_json = json.dumps(tokenizer_json)

    return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)


def safe_wer(reference: str, prediction: str) -> float:
    reference = reference.strip()
    prediction = prediction.strip()

    if not reference and not prediction:
        return 0.0
    if not reference and prediction:
        return 1.0
    return float(jiwer.wer(reference, prediction))


def calc_similarity(wer_value: float) -> float:
    # Maps WER [0..inf) into (0..1], where 1 is perfect.
    return 1.0 / (1.0 + wer_value)


def build_training_validation_indices(n_samples: int, val_fraction: float, seed: int) -> np.ndarray:
    """
    Reproduce the exact split logic from train-seq2seq.py:
    val_size = max(1, round(n_samples * val_fraction)), capped at n_samples - 1
    val_idx = shuffled_indices[:val_size] with np.random.default_rng(seed)
    """
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to build train/validation split.")

    val_size = max(1, int(round(n_samples * val_fraction)))
    val_size = min(val_size, n_samples - 1)

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_samples)
    rng.shuffle(all_idx)
    return all_idx[:val_size]


def build_custom_test_indices(n_samples: int, test_ratio: float, seed: int) -> np.ndarray:
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to build a custom test split.")

    ratio = min(max(test_ratio, 0.01), 0.9)
    test_size = max(1, int(round(n_samples * ratio)))
    test_size = min(test_size, n_samples - 1)

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_samples)
    rng.shuffle(all_idx)
    return all_idx[:test_size]


def choose_middle(entries: List[Dict[str, Any]], n_show: int) -> List[Dict[str, Any]]:
    if not entries:
        return []

    sorted_by_sim = sorted(entries, key=lambda e: e["similarity"], reverse=True)
    mid = len(sorted_by_sim) // 2

    half = n_show // 2
    start = max(0, mid - half)
    end = min(len(sorted_by_sim), start + n_show)
    start = max(0, end - n_show)
    return sorted_by_sim[start:end]


def print_header(model_path: Path, tokenizer_path: Path, data_path: Path, history_path: Path):
    print("=" * 88)
    print("SignAI v36 Evaluation")
    print("=" * 88)
    print(f"Model            : {model_path}")
    print(f"Tokenizer        : {tokenizer_path}")
    print(f"Data folder      : {data_path}")
    print(f"History image    : {history_path} ({'found' if history_path.exists() else 'not found'})")
    print("=" * 88)


def print_metrics_summary(entries: List[Dict[str, Any]]):
    mean_wer = float(np.mean([e["wer"] for e in entries]))
    median_wer = float(np.median([e["wer"] for e in entries]))
    exact_rate = 100.0 * float(np.mean([1.0 if e["exact_match"] else 0.0 for e in entries]))

    print("\n" + "=" * 88)
    print("Summary")
    print("=" * 88)
    print(f"Mean WER         : {mean_wer:.4f}")
    print(f"Median WER       : {median_wer:.4f}")
    print(f"Exact match rate : {exact_rate:.2f}%")


def print_block(title: str, rows: List[Dict[str, Any]], n: int):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    for i, row in enumerate(rows[:n], start=1):
        print(
            f"[{i}] idx={row['idx']} | WER={row['wer']:.3f} | Similarity={row['similarity']:.3f} | exact={row['exact_match']}"
        )
        print(f"    file : {row['source_file']}")
        print(f"    ref  : {row['reference']}")
        print(f"    pred : {row['prediction']}")
        print("-" * 88)


def load_v36_model_with_fallback(model_path: Path, train_module, encoder_input: np.ndarray, tokenizer):
    keras.config.enable_unsafe_deserialization()
    custom_objects = {
        "create_cross_mask": create_cross_mask,
        "<lambda>": encoder_mask_fn,
    }

    try:
        return tf.keras.models.load_model(
            str(model_path),
            compile=False,
            safe_mode=False,
            custom_objects=custom_objects,
        )
    except Exception as e:
        print(f"[WARN] Direct model load failed: {e}")
        print("[WARN] Falling back to architecture rebuild + load_weights...")

        vocab_size = len(tokenizer.word_index) + 1
        max_frames = int(encoder_input.shape[1])
        num_features = int(encoder_input.shape[2])

        model = train_module.build_seq2seq_model(
            max_frames=max_frames,
            num_features=num_features,
            vocab_size=vocab_size,
            embedding_dim=256,
            encoder_units=256,
            decoder_units=512,
            dropout_rate=0.4,
            recurrent_dropout_rate=0.1,
            architecture="multi_attention",
            use_layer_norm=True,
            use_multi_head_attention=True,
            num_attention_heads=8,
        )
        model.load_weights(str(model_path))
        return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate v36 model on test data and print best/middle/worst translations."
    )
    parser.add_argument("--data-folder", default=str(DEFAULT_DATA_FOLDER), help="Folder with CSV samples")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Directory with model files")
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER_PATH), help="Tokenizer JSON path")
    parser.add_argument("--history-image", default=str(DEFAULT_HISTORY_IMAGE), help="Path to training history image")
    parser.add_argument(
        "--split-source",
        choices=["training", "custom"],
        default="training",
        help="training = reproduce exact val split from train-seq2seq.py; custom = use --test-ratio",
    )
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Used only when --split-source custom")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic split")
    parser.add_argument("--max-decode-len", type=int, default=40, help="Max decoder steps per sentence")
    parser.add_argument("--show-n", type=int, default=5, help="How many sentences per group")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for evaluated test samples (0 = all)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    train_module = load_train_module(repo_root)

    models_dir = resolve_path(repo_root, args.models_dir)
    data_folder = resolve_path(repo_root, args.data_folder)
    tokenizer_path = resolve_path(repo_root, args.tokenizer)
    history_image = resolve_path(repo_root, args.history_image)

    model_path = find_latest_v36_model(models_dir)
    tokenizer = load_tokenizer(tokenizer_path)

    print_header(model_path, tokenizer_path, data_folder, history_image)

    samples = train_module.load_data_from_folder(str(data_folder), use_cache=True)
    encoder_input, _ = train_module.build_encoder_input(samples)

    model = load_v36_model_with_fallback(model_path, train_module, encoder_input, tokenizer)

    all_refs = [train_module.normalize_text(s.get("gloss", "")) for s in samples]
    if args.split_source == "training":
        test_indices = build_training_validation_indices(len(samples), val_fraction=0.1, seed=args.seed)
        split_label = "training validation split (exact recreation)"
    else:
        test_indices = build_custom_test_indices(len(samples), test_ratio=args.test_ratio, seed=args.seed)
        split_label = f"custom split (ratio={args.test_ratio})"

    if args.max_samples and args.max_samples > 0:
        test_indices = test_indices[: args.max_samples]

    entries: List[Dict[str, Any]] = []

    print(f"Total samples      : {len(samples)}")
    print(f"Split source       : {split_label}")
    print(f"Test samples used  : {len(test_indices)}")
    print(f"Max decode length  : {args.max_decode_len}")
    print("Running translations ...")

    for rank, idx in enumerate(test_indices, start=1):
        pred = train_module.greedy_decode(
            model,
            encoder_input[idx],
            tokenizer,
            max_len=int(args.max_decode_len),
        )
        pred_norm = train_module.normalize_text(pred)
        ref_norm = all_refs[idx]

        wer_value = safe_wer(ref_norm, pred_norm)
        similarity = calc_similarity(wer_value)

        entries.append(
            {
                "idx": int(idx),
                "order": int(rank),
                "source_file": samples[idx].get("source_file", ""),
                "reference": ref_norm,
                "prediction": pred_norm,
                "wer": wer_value,
                "similarity": similarity,
                "exact_match": ref_norm == pred_norm,
            }
        )

    if not entries:
        raise RuntimeError("No test predictions were generated.")

    sorted_best = sorted(entries, key=lambda e: e["similarity"], reverse=True)
    sorted_worst = sorted(entries, key=lambda e: e["similarity"])
    middle = choose_middle(entries, max(1, int(args.show_n)))

    print_metrics_summary(entries)

    n_show = max(1, int(args.show_n))
    print_block("BEST SENTENCES", sorted_best, n_show)
    print_block("MIDDLE SENTENCES", middle, n_show)
    print_block("WORST SENTENCES", sorted_worst, n_show)


if __name__ == "__main__":
    main()
