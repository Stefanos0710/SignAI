"""Evaluate a trained WordClassifier checkpoint on val/test: top-k accuracy, class-balanced
(macro) metrics, calibration, accuracy by class frequency / gloss category / clip duration,
most-confused gloss pairs, and optionally a per-stream permutation ablation and a train-subset
pass (generalization gap).

Run from the repo root:
    python signai/word_classification/models/evaluate.py                        # best ckpt, val + test
    python signai/word_classification/models/evaluate.py --checkpoint latest
    python signai/word_classification/models/evaluate.py --ablation             # + shuffle each stream (test)
    python signai/word_classification/models/evaluate.py --train-subset 30000   # + train-subset metrics

Inference runs once per split/variant: the logits are cached in checkpoints/eval/<checkpoint>/
(keyed by the checkpoint's mtime and size), so re-running only recomputes the metrics, not the
model. Writes into that folder: report.md, metrics.json, per_class_<split>.csv,
confusions_<split>.csv and the *.png plots.

Note on the split: preprocessing.py splits stratified per class but *not* per signer, so the
same signers/sessions occur in train and test -- the test numbers are within-signer numbers,
an upper bound for unseen signers (the report quantifies the overlap).
"""
import argparse
import csv
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.models.gpu_env import ensure_gpu_env  # noqa: E402
ensure_gpu_env()  # must run before `import tensorflow` -- see gpu_env.py

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402

from signai.word_classification.models.model import WordClassifier  # noqa: E402, F401 (registers the class)
from signai.word_classification.models.train import (  # noqa: E402
    BEST_CKPT, CHECKPOINT_DIR, LATEST_CKPT, LOG_FILE, load_split,
)
from signai.word_classification.models.hand_stream import load_hand_features  # noqa: E402
from signai.word_classification.models.face_stream import load_face_features  # noqa: E402
from signai.word_classification.paths import DATASET_DIR as RAW_DATASET_DIR, PROCESSED_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

EVAL_ROOT = CHECKPOINT_DIR / "eval"
SEGMENTS_CSV = RAW_DATASET_DIR / "segments.csv"
CLIPS_DIR = RAW_DATASET_DIR / "word_clips"

BATCH_SIZE = 32
TOP_KS = (1, 2, 3, 5, 10, 20, 50)
CURVE_KS = np.arange(1, 51)
ECE_BINS = 15
CONFIDENCE_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# train-set samples per class -> bucket label
FREQUENCY_BUCKETS = ((0, 10), (10, 20), (20, 50), (50, 100), (100, 500), (500, 1000), (1000, None))
DURATION_BUCKETS_MS = ((0, 100), (100, 160), (160, 240), (240, 360), (360, 640), (640, None))
MIN_SUPPORT_FOR_RANKING = 10  # best/worst class lists only consider classes with this many samples
# model input name -> streams it belongs to, for the permutation ablation (left/right share one
# permutation so the "hands" ablation keeps each clip's two hands together)
STREAMS = {"pose": ("keypoints",), "hands": ("left_hand", "right_hand"), "mouth": ("mouth",)}

# reference palette (dataviz skill, light mode)
COLOR_1, COLOR_2, COLOR_3 = "#2a78d6", "#eb6834", "#1baf7a"
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#fcfcfb"
BLUE_RAMP = ("#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b")


def gloss_category(gloss: str) -> str:
    """DGS-Korpus annotation conventions -> coarse gloss category. Everything that isn't one of
    the special (non-lexical) annotation types counts as "lexical"."""
    if gloss == "GEST" or gloss.startswith("GEST-"):
        return "gesture"
    if gloss == "INDEX" or gloss.startswith("INDEX-"):
        return "index (pointing)"
    if gloss == "PROD":
        return "productive"
    if gloss.startswith("ALPHA"):
        return "fingerspelling"
    if gloss == "NAME" or gloss.startswith("NAME-"):
        return "name"
    if gloss.startswith("LIST"):
        return "list"
    if gloss == "ORAL":
        return "oral"
    if "$KANDIDAT" in gloss:
        return "candidate ($KANDIDAT)"
    return "lexical"


# --- inference + cache -------------------------------------------------------------------------

def resolve_checkpoint(name: str) -> Path:
    return {"best": BEST_CKPT, "latest": LATEST_CKPT}.get(name, Path(name))


def checkpoint_key(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def load_inputs(split: str, indices=None):
    """The same four arrays train.py::build_dataset feeds the model, kept as numpy so they can be
    subset/permuted. Returns (inputs dict, y, clip_ids, classes)."""
    X, y, classes, _ = load_split(split, with_crops=False)
    clip_ids, left_hand, right_hand = load_hand_features(split)
    _, mouth = load_face_features(split)
    inputs = {"keypoints": X, "left_hand": left_hand, "right_hand": right_hand, "mouth": mouth}
    if indices is not None:
        inputs = {key: value[indices] for key, value in inputs.items()}
        y, clip_ids = y[indices], clip_ids[indices]
    return inputs, y, clip_ids, classes


def predict_logits(model, inputs, batch_size: int):
    dataset = tf.data.Dataset.from_tensor_slices(
        (inputs["keypoints"], inputs["left_hand"], inputs["right_hand"], inputs["mouth"])
    )
    # a 1-tuple, so keras' predict reads the 4-tuple as x (a bare 4-tuple would be parsed as x, y, ...)
    dataset = dataset.batch(batch_size).map(lambda k, l, r, m: ((k, l, r, m),)).prefetch(tf.data.AUTOTUNE)
    return model.predict(dataset, verbose=1).astype(np.float32)


class LogitCache:
    """Runs (or reuses cached) inference for one checkpoint. The model is loaded lazily, only if
    some split/variant actually needs recomputing."""

    def __init__(self, checkpoint: Path, out_dir: Path, batch_size: int, recompute: bool):
        self.checkpoint = checkpoint
        self.key = checkpoint_key(checkpoint)
        self.out_dir = out_dir
        self.batch_size = batch_size
        self.recompute = recompute
        self._model = None

    @property
    def model(self):
        if self._model is None:
            for gpu in tf.config.list_physical_devices("GPU"):
                # the GPU is shared -- don't grab all of its memory up front
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"loading {self.checkpoint}")
            self._model = keras.models.load_model(self.checkpoint)
        return self._model

    def get(self, split: str, variant: str = "", indices=None, permute: str | None = None, seed: int = 0):
        """-> (logits (N, C) float32, y (N,), clip_ids (N,), classes (C,))"""
        path = self.out_dir / f"{split}{'_' + variant if variant else ''}_logits.npz"
        if path.exists() and not self.recompute:
            with np.load(path, allow_pickle=True) as cached:
                if str(cached["checkpoint_key"]) == self.key:
                    logging.info(f"using cached logits {path.name}")
                    return cached["logits"], cached["y"], cached["clip_ids"], cached["classes"]

        inputs, y, clip_ids, classes = load_inputs(split, indices)
        if permute is not None:
            perm = np.random.default_rng(seed).permutation(len(y))
            for key in STREAMS[permute]:
                inputs[key] = inputs[key][perm]

        logging.info(f"{split}{' (' + variant + ')' if variant else ''}: predicting {len(y):,} clips")
        started = time.time()
        logits = predict_logits(self.model, inputs, self.batch_size)
        logging.info(f"done in {time.time() - started:.0f}s")
        np.savez(path, logits=logits, y=y, clip_ids=clip_ids, classes=classes, checkpoint_key=self.key)
        return logits, y, clip_ids, classes


# --- metrics -----------------------------------------------------------------------------------

def log_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def true_class_rank(logits, y):
    """0-based rank of the true class (0 == top-1 correct)."""
    true_logit = logits[np.arange(len(y)), y][:, None]
    return (logits > true_logit).sum(axis=1)


def expected_calibration_error(confidence, correct, n_bins: int = ECE_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    ece, mce, rows = 0.0, 0.0, []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            rows.append((edges[b], edges[b + 1], 0, None, None))
            continue
        acc, conf = correct[mask].mean(), confidence[mask].mean()
        ece += mask.mean() * abs(acc - conf)
        mce = max(mce, abs(acc - conf))
        rows.append((edges[b], edges[b + 1], int(mask.sum()), float(acc), float(conf)))
    return float(ece), float(mce), rows


def fit_temperature(logits, y, lo: float = 0.05, hi: float = 10.0, iters: int = 60):
    """Temperature T minimizing mean NLL of softmax(logits / T) -- golden-section search over
    log T (the NLL is unimodal in T)."""
    def nll(log_t):
        return -log_softmax(logits / np.exp(log_t))[np.arange(len(y)), y].mean()

    a, b = np.log(lo), np.log(hi)
    ratio = (np.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = nll(d)
    return float(np.exp((a + b) / 2))


def bucket_label(lo, hi, unit=""):
    """[lo, hi) -> "lo-hi ms" for durations, "lo-(hi-1)" for integer counts, ">= lo" if open."""
    if hi is None:
        return f">= {lo}{unit}"
    if lo == 0:
        return f"< {hi}{unit}"
    return f"{lo}-{hi}{unit}" if unit else f"{lo}-{hi - 1}"


def in_bucket(values, lo, hi):
    return (values >= lo) & (values < hi) if hi is not None else values >= lo


def group_metrics(mask, rank, y, num_classes):
    """Sample-level top-1/top-5 plus class-balanced (macro) top-1 for the samples in `mask`."""
    if not mask.any():
        return {"samples": 0}
    support = np.bincount(y[mask], minlength=num_classes)
    hits = np.bincount(y[mask], weights=(rank[mask] == 0), minlength=num_classes)
    present = support > 0
    return {
        "samples": int(mask.sum()),
        "classes": int(present.sum()),
        "top1": float((rank[mask] < 1).mean()),
        "top5": float((rank[mask] < 5).mean()),
        "macro_top1": float((hits[present] / support[present]).mean()),
    }


def compute_metrics(logits, y, classes, train_counts, temperature=None):
    """Everything report.md shows for one split. Returns (metrics dict, per-class rows,
    confusion rows, arrays for the plots)."""
    num_classes = len(classes)
    n = len(y)
    log_probs = log_softmax(logits)
    probs = np.exp(log_probs)
    rank = true_class_rank(logits, y)
    pred = logits.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = pred == y

    support = np.bincount(y, minlength=num_classes)
    present = support > 0

    def macro_topk(k):
        hits = np.bincount(y, weights=(rank < k), minlength=num_classes)
        return float((hits[present] / support[present]).mean())

    # frequency-prior baseline: always guess the k most frequent training classes
    prior_order = np.argsort(-train_counts, kind="stable")
    prior_rank = np.empty(num_classes, dtype=np.int64)
    prior_rank[prior_order] = np.arange(num_classes)

    metrics = {
        "samples": int(n),
        "classes_in_split": int(present.sum()),
        "loss": float(-log_probs[np.arange(n), y].mean()),
        "topk": {f"top{k}": float((rank < k).mean()) for k in TOP_KS},
        "macro_topk": {f"top{k}": macro_topk(k) for k in TOP_KS},
        "prior_baseline_topk": {f"top{k}": float((prior_rank[y] < k).mean()) for k in TOP_KS},
        "mrr": float((1.0 / (rank + 1)).mean()),
        "median_true_rank": float(np.median(rank + 1)),
        "mean_true_rank": float((rank + 1).mean()),
    }

    # precision/recall/F1 (top-1); classes never predicted get precision 0 (sklearn's
    # zero_division=0 convention)
    tp = np.bincount(y[correct], minlength=num_classes).astype(float)
    n_pred = np.bincount(pred, minlength=num_classes).astype(float)
    precision = np.divide(tp, n_pred, out=np.zeros(num_classes), where=n_pred > 0)
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=present)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_classes),
                   where=(precision + recall) > 0)
    metrics["macro"] = {
        "precision": float(precision[present].mean()),
        "recall_balanced_accuracy": float(recall[present].mean()),
        "f1": float(f1[present].mean()),
    }
    metrics["weighted"] = {
        "precision": float((precision * support).sum() / n),
        "recall": float((recall * support).sum() / n),
        "f1": float((f1 * support).sum() / n),
    }

    # prediction behaviour
    top10 = prior_order[:10]
    metrics["predictions"] = {
        "distinct_classes_predicted": int((n_pred > 0).sum()),
        "classes_in_split_never_predicted": int((present & (n_pred == 0)).sum()),
        "classes_in_split_never_correct": int((present & (tp == 0)).sum()),
        "share_true_in_top10_train_classes": float(np.isin(y, top10).mean()),
        "share_pred_in_top10_train_classes": float(np.isin(pred, top10).mean()),
    }

    # calibration + selective prediction
    ece, mce, reliability = expected_calibration_error(confidence, correct)
    metrics["calibration"] = {
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "ece": ece,
        "mce": mce,
        "mean_confidence_correct": float(confidence[correct].mean()),
        "mean_confidence_wrong": float(confidence[~correct].mean()),
    }
    reliability_scaled = None
    if temperature is not None:
        scaled_log_probs = log_softmax(logits / temperature)
        scaled_conf = np.exp(scaled_log_probs).max(axis=1)
        ece_t, mce_t, reliability_scaled = expected_calibration_error(scaled_conf, correct)
        metrics["calibration"]["temperature"] = temperature
        metrics["calibration"]["loss_after_temperature"] = float(-scaled_log_probs[np.arange(n), y].mean())
        metrics["calibration"]["ece_after_temperature"] = ece_t
        metrics["calibration"]["mce_after_temperature"] = mce_t

    metrics["selective"] = [
        {"threshold": t, "coverage": float((confidence >= t).mean()),
         "accuracy": float(correct[confidence >= t].mean()) if (confidence >= t).any() else None}
        for t in CONFIDENCE_THRESHOLDS
    ]
    order = np.argsort(-confidence, kind="stable")
    cumulative_acc = np.cumsum(correct[order]) / np.arange(1, n + 1)
    metrics["accuracy_at_coverage"] = {
        f"{int(c * 100)}%": float(cumulative_acc[max(int(round(c * n)) - 1, 0)])
        for c in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    }

    # breakdowns
    class_train_count = train_counts[y]
    metrics["by_train_frequency"] = {
        bucket_label(lo, hi): group_metrics(in_bucket(class_train_count, lo, hi), rank, y, num_classes)
        for lo, hi in FREQUENCY_BUCKETS
    }
    categories = np.array([gloss_category(c) for c in classes])
    sample_category = categories[y]
    metrics["by_category"] = {
        cat: group_metrics(sample_category == cat, rank, y, num_classes)
        for cat, _ in Counter(sample_category).most_common()
    }

    # confusions (true != pred), most frequent first
    off = ~correct
    pair_ids, pair_counts = np.unique(y[off] * num_classes + pred[off], return_counts=True)
    order = np.argsort(-pair_counts, kind="stable")[:50]
    confusion_rows = [
        {"true": classes[pid // num_classes], "predicted": classes[pid % num_classes], "count": int(cnt),
         "share_of_true_class": float(cnt / support[pid // num_classes])}
        for pid, cnt in zip(pair_ids[order], pair_counts[order])
    ]

    # per-class table
    most_confused = {}
    for pid, cnt in zip(pair_ids, pair_counts):
        t = pid // num_classes
        if cnt > most_confused.get(t, (None, 0))[1]:
            most_confused[t] = (pid % num_classes, cnt)
    top5_hits = np.bincount(y, weights=(rank < 5), minlength=num_classes)
    mean_conf_true = np.bincount(y, weights=probs[np.arange(n), y], minlength=num_classes)
    per_class = []
    for c in np.where(present)[0]:
        confused_with, confused_count = most_confused.get(c, (None, 0))
        per_class.append({
            "class": classes[c],
            "category": categories[c],
            "train_count": int(train_counts[c]),
            "support": int(support[c]),
            "top1_recall": float(recall[c]),
            "top5_recall": float(top5_hits[c] / support[c]),
            "precision": float(precision[c]),
            "f1": float(f1[c]),
            "times_predicted": int(n_pred[c]),
            "mean_prob_true_class": float(mean_conf_true[c] / support[c]),
            "most_confused_with": classes[confused_with] if confused_with is not None else "",
            "most_confused_count": int(confused_count),
        })

    ranked = [row for row in per_class if row["support"] >= MIN_SUPPORT_FOR_RANKING]
    ranked.sort(key=lambda row: (row["f1"], row["support"]))
    metrics["worst_classes"] = ranked[:15]
    metrics["best_classes"] = ranked[::-1][:15]

    curves = {
        "topk_micro": np.array([(rank < k).mean() for k in CURVE_KS]),
        "topk_macro": np.array([macro_topk(k) for k in CURVE_KS]),
        "topk_prior": np.array([(prior_rank[y] < k).mean() for k in CURVE_KS]),
        "reliability": reliability,
        "reliability_scaled": reliability_scaled,
        "coverage_accuracy": cumulative_acc,
        "rank": rank,
        "pred": pred,
    }
    return metrics, per_class, confusion_rows, curves


def duration_metrics(clip_ids, rank, y, classes, num_classes):
    """Accuracy by clip duration, via clip_id -> clip filename -> segments.csv row. clip_id is the
    clip's position in preprocessing.list_jobs()'s sorted list; reconstructed here the same way
    and verified against the stored labels, so a changed word_clips/ folder is caught, not
    silently misattributed. Returns (metrics or None, sessions per clip_id)."""
    rows = {}
    with open(SEGMENTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            transcript = row["transcript"].strip()
            if not transcript:
                continue
            name = f"{Path(row['eaf_file']).stem}_{row['participant']}_{int(row['index']):04d}.mp4"
            rows[name] = (transcript, int(row["end_ms"]) - int(row["start_ms"]),
                          f"{Path(row['eaf_file']).stem}_{row['participant']}")
    names = [p.name for p in sorted(CLIPS_DIR.glob("*.mp4")) if p.name in rows]

    if clip_ids.max() >= len(names) or any(rows[names[cid]][0] != classes[label]
                                           for cid, label in zip(clip_ids, y)):
        logging.warning("clip_id -> word_clips mapping doesn't match the stored labels, "
                        "skipping duration/session analysis")
        return None, None

    durations = np.array([rows[names[cid]][1] for cid in clip_ids])
    by_duration = {
        bucket_label(lo, hi, " ms"): group_metrics(in_bucket(durations, lo, hi), rank, y, num_classes)
        for lo, hi in DURATION_BUCKETS_MS
    }
    session_of = {cid: rows[name][2] for cid, name in enumerate(names)}
    return by_duration, session_of


# --- plots -------------------------------------------------------------------------------------

def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)


def make_plots(out_dir: Path, split: str, metrics, curves, classes, y, train_counts, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 9})

    # 1) top-k curve
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = (
        (curves["topk_micro"], COLOR_1, "model (per clip)", "-"),
        (curves["topk_macro"], COLOR_2, "model (per class, balanced)", "-"),
        (curves["topk_prior"], COLOR_3, "baseline (per clip): k most frequent glosses", "--"),
    )
    label_y = {}  # end-of-line labels, pushed apart so close curves don't overprint
    for i in np.argsort([values[-1] for values, *_ in series]):
        y_end = series[i][0][-1] * 100
        label_y[i] = max(y_end, max(label_y.values(), default=-10) + 4)
    for i, (values, color, label, style) in enumerate(series):
        ax.plot(CURVE_KS, values * 100, style, color=color, linewidth=2, label=label)
        ax.annotate(f"{values[-1] * 100:.1f}%", (CURVE_KS[-1], label_y[i]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=TEXT_PRIMARY, fontsize=8)
    for k in (1, 5, 10):
        ax.plot(k, curves["topk_micro"][k - 1] * 100, "o", color=COLOR_1, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"top-{k}: {curves['topk_micro'][k - 1] * 100:.1f}%", (k, curves["topk_micro"][k - 1] * 100),
                    xytext=(6, -12), textcoords="offset points", color=TEXT_PRIMARY, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50], labels=["1", "2", "3", "5", "10", "20", "50"])
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper left", labelcolor=TEXT_PRIMARY)
    _style(ax, f"Top-k accuracy ({split}, {len(classes)} glosses)", "k", "accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_dir / f"topk_{split}.png", dpi=150)
    plt.close(fig)

    # 2) accuracy by training-set frequency
    buckets = metrics["by_train_frequency"]
    labels = [label for label, m in buckets.items() if m["samples"]]
    top1 = [buckets[label]["top1"] * 100 for label in labels]
    top5 = [buckets[label]["top5"] * 100 for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.38
    ax.bar(x - width / 2 - 0.01, top1, width, color=COLOR_1, label="top-1")
    ax.bar(x + width / 2 + 0.01, top5, width, color=COLOR_2, label="top-5")
    for xi, v1, v5 in zip(x, top1, top5):
        ax.annotate(f"{v1:.0f}", (xi - width / 2, v1), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7, color=TEXT_PRIMARY)
        ax.annotate(f"{v5:.0f}", (xi + width / 2, v5), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7, color=TEXT_PRIMARY)
    ax.set_xticks(x, labels=[f"{label}\n{buckets[label]['classes']} cl. / {buckets[label]['samples']:,} clips"
                             for label in labels], fontsize=7)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    _style(ax, f"Accuracy by training samples per gloss ({split})", "training clips per gloss", "accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_dir / f"accuracy_by_frequency_{split}.png", dpi=150)
    plt.close(fig)

    # 3) reliability diagram
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle=":", label="perfectly calibrated")
    for rows, color, label in ((curves["reliability"], COLOR_1, "raw"),
                               (curves["reliability_scaled"], COLOR_2, "temperature-scaled")):
        if rows is None:
            continue
        pts = [(conf, acc) for _, _, count, acc, conf in rows if count]
        ax.plot(*zip(*pts), "-o", color=color, linewidth=2, markersize=5, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left", labelcolor=TEXT_PRIMARY)
    cal = metrics["calibration"]
    subtitle = f"ECE {cal['ece']:.3f}"
    if "ece_after_temperature" in cal:
        subtitle += f" -> {cal['ece_after_temperature']:.3f} (T={cal['temperature']:.2f})"
    _style(ax, f"Calibration ({split}): {subtitle}", "mean confidence (top-1 prob.)", "accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / f"reliability_{split}.png", dpi=150)
    plt.close(fig)

    # 4) accuracy vs coverage (answer only when confident)
    acc = curves["coverage_accuracy"]
    coverage = np.arange(1, len(acc) + 1) / len(acc)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(coverage * 100, acc * 100, color=COLOR_1, linewidth=2)
    for c in (0.25, 0.5, 0.75):
        i = int(round(c * len(acc))) - 1
        ax.plot(c * 100, acc[i] * 100, "o", color=COLOR_1, markersize=6, markeredgecolor=SURFACE)
        ax.annotate(f"{acc[i] * 100:.1f}% @ {c * 100:.0f}%", (c * 100, acc[i] * 100), xytext=(6, 4),
                    textcoords="offset points", color=TEXT_PRIMARY, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    _style(ax, f"Selective prediction ({split}): most-confident x% of clips",
           "coverage (% of clips answered)", "top-1 accuracy on answered clips (%)")
    fig.tight_layout()
    fig.savefig(out_dir / f"coverage_{split}.png", dpi=150)
    plt.close(fig)

    # 5) confusion matrix of the 30 most frequent glosses (row-normalized)
    top = np.argsort(-train_counts, kind="stable")[:30]
    index = {c: i for i, c in enumerate(top)}
    matrix = np.zeros((len(top), len(top) + 1))
    for t, p in zip(y, curves["pred"]):
        if t in index:
            matrix[index[t], index.get(p, len(top))] += 1
    matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    names = [classes[c] for c in top]
    ax.set_xticks(range(len(top) + 1), labels=names + ["(other)"], rotation=90, fontsize=7, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(top)), labels=names, fontsize=7, color=TEXT_SECONDARY)
    ax.set_xlabel("predicted", color=TEXT_SECONDARY)
    ax.set_ylabel("true", color=TEXT_SECONDARY)
    ax.set_title(f"Confusions among the 30 most frequent glosses ({split}, row-normalized)",
                 color=TEXT_PRIMARY, fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_top30_{split}.png", dpi=150)
    plt.close(fig)

    # 6) training curves (from train.log), once
    if history and not (out_dir / "training_curves.png").exists():
        epochs = [h["epoch"] for h in history]
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
        for ax, key, title in ((ax_loss, "loss", "Loss"), (ax_acc, "acc", "Accuracy")):
            scale = 100 if key == "acc" else 1
            ax.plot(epochs, [h[key] * scale for h in history], color=COLOR_1, linewidth=2, label="train")
            ax.plot(epochs, [h["val_" + key] * scale for h in history], color=COLOR_2, linewidth=2, label="val")
            best = max(history, key=lambda h: h["val_acc"])
            ax.axvline(best["epoch"], color=TEXT_SECONDARY, linestyle=":", linewidth=1)
            ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
            _style(ax, f"{title} per epoch (best val epoch {best['epoch']})", "epoch",
                   "accuracy (%)" if key == "acc" else "cross-entropy")
        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=150)
        plt.close(fig)


def parse_train_log(path: Path = LOG_FILE):
    pattern = re.compile(r"epoch (\d+)/\d+ done .*? loss ([\d.]+) acc ([\d.]+) \| val_loss ([\d.]+) val_acc ([\d.]+)")
    history = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                history[epoch] = {"epoch": epoch, "loss": float(m.group(2)), "acc": float(m.group(3)),
                                  "val_loss": float(m.group(4)), "val_acc": float(m.group(5))}
    return [history[e] for e in sorted(history)]


# --- report ------------------------------------------------------------------------------------

def pct(value, digits=1):
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def write_csv(path: Path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_table(groups, first_col):
    lines = [f"| {first_col} | glosses | clips | top-1 | top-5 | macro top-1 |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in groups.items():
        if m["samples"]:
            lines.append(f"| {name} | {m['classes']:,} | {m['samples']:,} | {pct(m['top1'])} | "
                         f"{pct(m['top5'])} | {pct(m['macro_top1'])} |")
    return lines


def render_report(results, checkpoint: Path, history):
    lines = [f"# WordClassifier evaluation: `{checkpoint.name}`", ""]
    if history:
        best = max(history, key=lambda h: h["val_acc"])
        last = history[-1]
        lines += [
            f"Training: {len(history)} epochs, best val_acc {pct(best['val_acc'], 2)} at epoch {best['epoch']}; "
            f"last epoch train acc {pct(last['acc'], 2)} (with dropout) / val acc {pct(last['val_acc'], 2)}, "
            f"val_loss lowest at epoch {min(history, key=lambda h: h['val_loss'])['epoch']} "
            f"({min(h['val_loss'] for h in history):.3f}), {last['val_loss']:.3f} at the end.",
            "",
        ]

    splits = [s for s in ("test", "val", "train") if s in results]
    lines += ["## Overview", "", "| metric | " + " | ".join(splits) + " |", "|---|" + "---:|" * len(splits)]

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(results[s]["metrics"]) for s in splits) + " |")

    row("clips", lambda m: f"{m['samples']:,}")
    row("glosses present", lambda m: f"{m['classes_in_split']:,}")
    for k in TOP_KS:
        row(f"top-{k} accuracy", lambda m, k=k: pct(m["topk"][f"top{k}"], 2))
    for k in (1, 5, 10):
        row(f"macro top-{k} (class-balanced)", lambda m, k=k: pct(m["macro_topk"][f"top{k}"], 2))
    row("macro precision", lambda m: pct(m["macro"]["precision"], 2))
    row("macro F1", lambda m: pct(m["macro"]["f1"], 2))
    row("weighted F1", lambda m: pct(m["weighted"]["f1"], 2))
    row("MRR", lambda m: f"{m['mrr']:.4f}")
    row("median rank of true gloss", lambda m: f"{m['median_true_rank']:.0f}")
    row("mean rank of true gloss", lambda m: f"{m['mean_true_rank']:.1f}")
    row("cross-entropy", lambda m: f"{m['loss']:.4f}")
    row("ECE", lambda m: f"{m['calibration']['ece']:.4f}")
    row("baseline top-1 (always most frequent)", lambda m: pct(m["prior_baseline_topk"]["top1"], 2))
    row("baseline top-5 (5 most frequent)", lambda m: pct(m["prior_baseline_topk"]["top5"], 2))
    lines.append("")

    test = results["test"]["metrics"]
    lines += ["![top-k](topk_test.png)", ""]

    if "ablation" in results:
        lines += ["## Stream importance (test, permutation ablation)", "",
                  "Each stream's inputs are shuffled across clips (same distribution, wrong clip), "
                  "the others stay intact. Bigger drop = the model relies more on that stream.", "",
                  "| shuffled stream | top-1 | Δ top-1 | top-5 | Δ top-5 | macro top-1 |",
                  "|---|---:|---:|---:|---:|---:|",
                  f"| none | {pct(test['topk']['top1'], 2)} | | {pct(test['topk']['top5'], 2)} | | "
                  f"{pct(test['macro_topk']['top1'], 2)} |"]
        for stream, m in results["ablation"].items():
            lines.append(
                f"| {stream} | {pct(m['top1'], 2)} | {(m['top1'] - test['topk']['top1']) * 100:+.2f} pp | "
                f"{pct(m['top5'], 2)} | {(m['top5'] - test['topk']['top5']) * 100:+.2f} pp | {pct(m['macro_top1'], 2)} |")
        lines.append("")

    if "train" in results:
        tr = results["train"]["metrics"]
        lines += ["## Generalization gap", "",
                  f"Random {tr['samples']:,}-clip train subset, evaluated without dropout: top-1 "
                  f"{pct(tr['topk']['top1'], 2)} vs test {pct(test['topk']['top1'], 2)} "
                  f"({(tr['topk']['top1'] - test['topk']['top1']) * 100:+.2f} pp), loss {tr['loss']:.3f} vs "
                  f"{test['loss']:.3f}.", ""]

    pred = test["predictions"]
    lines += [
        "## Prediction behaviour (test)", "",
        f"- distinct glosses predicted at least once: {pred['distinct_classes_predicted']:,} of {len(results['classes']):,}",
        f"- glosses in test never predicted: {pred['classes_in_split_never_predicted']:,}; "
        f"never predicted correctly: {pred['classes_in_split_never_correct']:,} "
        f"({pred['classes_in_split_never_correct'] / test['classes_in_split'] * 100:.1f}% of glosses in test)",
        f"- share of test clips whose true gloss is one of the 10 most frequent training glosses: "
        f"{pct(pred['share_true_in_top10_train_classes'])}; share of *predictions* that are one of them: "
        f"{pct(pred['share_pred_in_top10_train_classes'])}",
        "",
        "## Accuracy by training-set frequency of the gloss (test)", "",
        *group_table(test["by_train_frequency"], "train clips per gloss"),
        "", "![frequency](accuracy_by_frequency_test.png)", "",
        "## Accuracy by gloss category (test)", "",
        "DGS-Korpus annotation types: `GEST*` gestures, `INDEX*` pointing, `PROD` productive signs, "
        "`ALPHA*` fingerspelling, `NAME*`, `LIST*`, `ORAL`, `$KANDIDAT` candidate signs; everything "
        "else counts as lexical.", "",
        *group_table(test["by_category"], "category"), "",
    ]
    if "by_duration" in results:
        lines += ["## Accuracy by clip duration (test)", "", *group_table(results["by_duration"], "duration"), ""]
    if "session_overlap" in results:
        lines += [f"Signer/session overlap: {pct(results['session_overlap'])} of test clips come from a "
                  "session+participant that also has clips in train (the split is per gloss, not per signer), "
                  "so these are within-signer numbers.", ""]

    cal = test["calibration"]
    lines += [
        "## Calibration and selective prediction (test)", "",
        f"- mean confidence {cal['mean_confidence']:.3f} vs accuracy {cal['accuracy']:.3f}; "
        f"ECE {cal['ece']:.4f}, MCE {cal['mce']:.4f}",
        f"- mean confidence when correct {cal['mean_confidence_correct']:.3f}, when wrong {cal['mean_confidence_wrong']:.3f}",
    ]
    if "temperature" in cal:
        lines.append(f"- temperature scaling fitted on val: T = {cal['temperature']:.3f} -> test loss "
                     f"{test['loss']:.4f} -> {cal['loss_after_temperature']:.4f}, ECE {cal['ece']:.4f} -> "
                     f"{cal['ece_after_temperature']:.4f} (accuracy unchanged)")
    lines += ["", "| confidence ≥ | coverage | top-1 on covered |", "|---:|---:|---:|"]
    for s in test["selective"]:
        lines.append(f"| {s['threshold']:.1f} | {pct(s['coverage'])} | {pct(s['accuracy'])} |")
    lines += ["", "| most-confident share answered | top-1 |", "|---:|---:|"]
    for cov, acc in test["accuracy_at_coverage"].items():
        lines.append(f"| {cov} | {pct(acc)} |")
    lines += ["", "![reliability](reliability_test.png) ![coverage](coverage_test.png)", ""]

    lines += ["## Most confused gloss pairs (test)", "", "| true | predicted | clips | share of true gloss |",
              "|---|---|---:|---:|"]
    for c in results["test"]["confusions"][:25]:
        lines.append(f"| {c['true']} | {c['predicted']} | {c['count']} | {pct(c['share_of_true_class'])} |")
    lines += ["", "![confusions](confusion_top30_test.png)", ""]

    for title, key in (("Best", "best_classes"), ("Worst", "worst_classes")):
        lines += [f"## {title} glosses by F1 (test, ≥ {MIN_SUPPORT_FOR_RANKING} test clips)", "",
                  "| gloss | train | test | top-1 recall | top-5 recall | precision | F1 | most confused with |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for c in test[key]:
            lines.append(f"| {c['class']} | {c['train_count']:,} | {c['support']} | {pct(c['top1_recall'])} | "
                         f"{pct(c['top5_recall'])} | {pct(c['precision'])} | {pct(c['f1'])} | "
                         f"{c['most_confused_with']} ({c['most_confused_count']}) |")
        lines.append("")

    if history:
        lines += ["## Training curves", "", "![training](training_curves.png)", ""]
    lines += ["Full per-gloss table: `per_class_test.csv`; top confusions: `confusions_test.csv`; "
              "raw numbers: `metrics.json`."]
    return "\n".join(lines) + "\n"


def main(checkpoint_name="best", splits=("val", "test"), batch_size=BATCH_SIZE, ablation=False,
         train_subset=0, recompute=False):
    checkpoint = resolve_checkpoint(checkpoint_name)
    out_dir = EVAL_ROOT / checkpoint.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = LogitCache(checkpoint, out_dir, batch_size, recompute)

    with np.load(PROCESSED_DIR / "train_data.npz", allow_pickle=True) as train_data:
        train_y, classes, train_clip_ids = train_data["y"], train_data["classes"], train_data["clip_ids"]
    train_counts = np.bincount(train_y, minlength=len(classes))
    history = parse_train_log()

    # val first: its logits fit the temperature used for test's calibration numbers
    temperature = None
    results = {"classes": classes.tolist()}
    for split in [s for s in ("val", "test") if s in splits]:
        logits, y, clip_ids, split_classes = cache.get(split)
        assert np.array_equal(split_classes, classes), f"{split}: class list differs from train's"
        if split == "val":
            temperature = fit_temperature(logits, y)
            logging.info(f"temperature fitted on val: {temperature:.3f}")
        metrics, per_class, confusions, curves = compute_metrics(
            logits, y, classes, train_counts, temperature if split == "test" else None)
        results[split] = {"metrics": metrics, "confusions": confusions}
        write_csv(out_dir / f"per_class_{split}.csv", per_class)
        write_csv(out_dir / f"confusions_{split}.csv", confusions)
        make_plots(out_dir, split, metrics, curves, classes, y, train_counts, history)
        logging.info(f"{split}: top-1 {metrics['topk']['top1']:.4f}  top-5 {metrics['topk']['top5']:.4f}  "
                     f"macro top-1 {metrics['macro_topk']['top1']:.4f}")

        if split == "test":
            by_duration, session_of = duration_metrics(clip_ids, curves["rank"], y, classes, len(classes))
            if by_duration is not None:
                results["by_duration"] = by_duration
                train_sessions = {session_of[c] for c in train_clip_ids}
                results["session_overlap"] = float(np.mean([session_of[c] in train_sessions for c in clip_ids]))

            if ablation:
                results["ablation"] = {}
                for stream in STREAMS:
                    logits_p, y_p, _, _ = cache.get("test", f"perm-{stream}", permute=stream)
                    m, _, _, _ = compute_metrics(logits_p, y_p, classes, train_counts)
                    results["ablation"][stream] = {"top1": m["topk"]["top1"], "top5": m["topk"]["top5"],
                                                   "macro_top1": m["macro_topk"]["top1"], "loss": m["loss"]}

    if train_subset:
        indices = np.sort(np.random.default_rng(0).choice(len(train_y), size=min(train_subset, len(train_y)),
                                                          replace=False))
        logits, y, _, _ = cache.get("train", f"subset{len(indices)}", indices=indices)
        metrics, _, _, _ = compute_metrics(logits, y, classes, train_counts)
        results["train"] = {"metrics": metrics}

    (out_dir / "metrics.json").write_text(json.dumps(
        {"checkpoint": str(checkpoint), **{k: v for k, v in results.items() if k != "classes"}},
        indent=2, ensure_ascii=False))
    if "test" in results:
        (out_dir / "report.md").write_text(render_report(results, checkpoint, history), encoding="utf-8")
    logging.info(f"wrote {out_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="best", help="best, latest, or a path to a .keras file")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--ablation", action="store_true",
                        help="also evaluate test with each stream (pose/hands/mouth) shuffled across clips")
    parser.add_argument("--train-subset", type=int, default=0, metavar="N",
                        help="also evaluate N random training clips (generalization gap)")
    parser.add_argument("--recompute", action="store_true", help="ignore cached logits")
    args = parser.parse_args()

    main(args.checkpoint, tuple(args.splits), args.batch_size, args.ablation, args.train_subset, args.recompute)
