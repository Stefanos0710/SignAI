"""Turn the per-word clips cut by `segmentation_videos.py` into a train/val/test dataset.

Input:
    dataset/segments.csv   -- one row per segment: eaf_file, participant, index, ..., transcript
    dataset/word_clips/    -- <eaf_stem>_<participant>_<index:04d>.mp4, cut from those rows

Output:
    dataset/processed/{train,val,test}_data.npz  -- X (N, MAX_FRAMES, 147) float32, y (N,) int64,
                                                    classes (C,) label strings
    dataset/processed/.keypoint_cache.npz        -- extracted sequences, so a re-split is free

Pipeline (mirrors signai/letter_classification/preprocess_v3.py, but per video instead of per image):
    1) Extract pose + hand keypoints per frame (MediaPipe, same settings as the shared trainer
       preprocessing in signai/preprocessing/train_data.py).
    2) Center on the shoulder midpoint and scale by shoulder distance -- using the *video* average
       shoulders, not per frame, so the signer's own movement stays in the signal.
    3) Interpolate landmarks MediaPipe missed, then Savitzky-Golay smooth over time.
    4) Resample / zero-pad every clip to MAX_FRAMES frames.
    5) Drop classes with fewer than MIN_SAMPLES_PER_CLASS clips, then split per class (stratified).

Run from the repo root:  python signai/word_classification/preprocessing.py [--rebuild-cache]
"""

import argparse
import csv
import logging
import multiprocessing
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

# repo root on sys.path so the shared preprocessing helpers can be imported when this file is run
# as a script (sys.path[0] is this folder, not the repo root)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from signai.preprocessing import train_data  # noqa: E402
from signai.preprocessing.train_data import (  # noqa: E402
    POSE_LANDMARKS,
    apply_temporal_savgol_smoothing,
    center_keypoints,
    extract_frames,
    extract_hand_keypoints,
    extract_pose_keypoints,
    interpolate_missing_keypoints,
    normalize_keypoints,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
CLIPS_DIR = DATASET_DIR / "word_clips"
SEGMENTS_CSV = DATASET_DIR / "segments.csv"
OUT_DIR = DATASET_DIR / "processed"
CACHE_FILE = OUT_DIR / ".keypoint_cache.npz"

N_POSE = len(POSE_LANDMARKS)
N_HAND = 21
N_LANDMARKS = N_POSE + 2 * N_HAND          # 7 pose + 42 hand = 49
N_FEATURES = N_LANDMARKS * 3               # 147 per frame (no face -- words are hand/arm shapes)
MAX_FRAMES = 32                            # ~99th percentile of the segment durations at 50 fps
MIN_SAMPLES_PER_CLASS = 5
NUM_WORKERS = 12
SEED = 42


def load_labels():
    """Map clip filename -> transcript, for the clips that actually exist on disk."""
    labels = {}
    with open(SEGMENTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            transcript = row["transcript"].strip()
            if not transcript:
                continue
            prefix = Path(row["eaf_file"]).stem
            name = f"{prefix}_{row['participant']}_{int(row['index']):04d}.mp4"
            labels[name] = transcript
    return labels


def average_shoulders(pose_keypoints):
    """Video-wide average left/right shoulder position (indices 1 and 2 of POSE_LANDMARKS)."""
    detected = [np.array(kp, dtype=float) for _, kp in pose_keypoints if kp]
    if not detected:
        return None, None
    stacked = np.stack(detected)
    return stacked[:, 1].mean(axis=0), stacked[:, 2].mean(axis=0)


def resample_or_pad(sequence, max_frames=MAX_FRAMES):
    """Fix the clip to `max_frames`: uniformly subsample if longer, zero-pad at the end if shorter."""
    seq = np.asarray(sequence, dtype=np.float32)
    n = seq.shape[0]
    if n > max_frames:
        return seq[np.linspace(0, n - 1, max_frames).round().astype(int)]
    if n < max_frames:
        return np.concatenate([seq, np.zeros((max_frames - n, seq.shape[1]), dtype=np.float32)])
    return seq


def extract_clip(job):
    """Worker: one clip -> (clip_name, label, (MAX_FRAMES, N_FEATURES) array) or a failure reason."""
    clip_path, label = job
    frames = extract_frames(str(clip_path))
    if not frames:
        return clip_path.name, label, None, "read_failed"

    pose_kp = extract_pose_keypoints(frames)
    hand_kp = extract_hand_keypoints(frames)
    avg_left, avg_right = average_shoulders(pose_kp)
    if avg_left is None:
        return clip_path.name, label, None, "no_pose"

    sequence = []
    for (_, pose), (_, left, right) in zip(pose_kp, hand_kp):
        pose_arr = np.array(pose, dtype=float) if pose else np.zeros((N_POSE, 3))
        left_arr = np.array(left, dtype=float) if left else np.zeros((N_HAND, 3))
        right_arr = np.array(right, dtype=float) if right else np.zeros((N_HAND, 3))

        all_kp = np.concatenate([pose_arr, left_arr, right_arr], axis=0)
        if pose:
            all_kp = center_keypoints(all_kp, avg_left, avg_right)
            all_kp = normalize_keypoints(all_kp, avg_left, avg_right)
        sequence.append(all_kp)

    sequence = interpolate_missing_keypoints(sequence)
    sequence = apply_temporal_savgol_smoothing(sequence, window_length=9, polyorder=2)
    flat = sequence.reshape(len(sequence), N_FEATURES)
    return clip_path.name, label, resample_or_pad(flat), None


def init_worker():
    train_data.init_worker()
    logging.getLogger().setLevel(logging.WARNING)  # one INFO line per clip is 9k lines of noise


def build_dataset(jobs, workers=NUM_WORKERS):
    """Extract every clip in parallel. Returns X (N, MAX_FRAMES, N_FEATURES) and the label strings."""
    X, labels = [], []
    failed = Counter()

    with multiprocessing.Pool(processes=workers, initializer=init_worker) as pool:
        for name, label, features, reason in tqdm(
            pool.imap_unordered(extract_clip, jobs, chunksize=8), total=len(jobs), desc="clips"
        ):
            if features is None:
                failed[reason] += 1
                logging.debug(f"skipping {name}: {reason}")
                continue
            X.append(features)
            labels.append(label)

    for reason, count in failed.items():
        logging.info(f"Failed ({reason}): {count} ({count / len(jobs) * 100:.2f}%)")
    return np.array(X, dtype=np.float32), np.array(labels)


def filter_rare_classes(X, labels, min_samples=MIN_SAMPLES_PER_CLASS):
    counts = Counter(labels)
    keep = np.array([counts[l] >= min_samples for l in labels])
    dropped_classes = sum(1 for c in counts.values() if c < min_samples)
    logging.info(
        f"Dropped {dropped_classes} class(es) with < {min_samples} samples "
        f"({(~keep).sum()} clips); {len(counts) - dropped_classes} classes left"
    )
    return X[keep], labels[keep]


def split_and_save(X, labels, output_folder=OUT_DIR, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    """Stratified split -- every class keeps at least one val and one test sample."""
    classes = np.array(sorted(set(labels.tolist())))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[l] for l in labels], dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in range(len(classes)):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(test_ratio * len(idx))))
        n_val = max(1, int(round(val_ratio * len(idx))))
        test_idx += idx[:n_test].tolist()
        val_idx += idx[n_test:n_test + n_val].tolist()
        train_idx += idx[n_test + n_val:].tolist()

    output_folder.mkdir(parents=True, exist_ok=True)
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        idx = np.array(idx)
        rng.shuffle(idx)
        np.savez_compressed(
            output_folder / f"{split_name}_data.npz", X=X[idx], y=y[idx], classes=classes
        )
        logging.info(f"{split_name}: {len(idx)} samples -> {output_folder / f'{split_name}_data.npz'}")

    logging.info(f"Shape per sample: ({MAX_FRAMES}, {N_FEATURES}) | classes: {len(classes)}")


def demo():
    """Self-check: python signai/word_classification/preprocessing.py --self-check"""
    short = np.ones((5, N_FEATURES), dtype=np.float32)
    padded = resample_or_pad(short)
    assert padded.shape == (MAX_FRAMES, N_FEATURES)
    assert padded[:5].all() and not padded[5:].any(), "padding must be zeros after the real frames"

    long = np.arange(100 * N_FEATURES, dtype=np.float32).reshape(100, N_FEATURES)
    sub = resample_or_pad(long)
    assert sub.shape == (MAX_FRAMES, N_FEATURES)
    assert np.array_equal(sub[0], long[0]) and np.array_equal(sub[-1], long[-1]), "keeps clip ends"

    labels = np.array(["A"] * 20 + ["B"] * 5 + ["C"] * 2)
    X = np.zeros((len(labels), MAX_FRAMES, N_FEATURES), dtype=np.float32)
    X_kept, labels_kept = filter_rare_classes(X, labels)
    assert set(labels_kept.tolist()) == {"A", "B"} and len(X_kept) == 25

    names = load_labels()
    existing = [p.name for p in CLIPS_DIR.glob("*.mp4")][:50] if CLIPS_DIR.exists() else []
    for name in existing:
        assert name in names, f"clip {name} has no row in {SEGMENTS_CSV.name}"
    print(f"self-check passed ({len(existing)} real clip name(s) checked)")


def main(rebuild_cache=False):
    if CACHE_FILE.exists() and not rebuild_cache:
        logging.info(f"Loading cached keypoints from {CACHE_FILE} (--rebuild-cache to re-extract)")
        cached = np.load(CACHE_FILE, allow_pickle=False)
        X, labels = cached["X"], cached["labels"]
    else:
        names = load_labels()
        jobs = [(p, names[p.name]) for p in sorted(CLIPS_DIR.glob("*.mp4")) if p.name in names]
        missing = len(list(CLIPS_DIR.glob("*.mp4"))) - len(jobs)
        if missing:
            logging.warning(f"{missing} clip(s) without a segments.csv row -- skipped")
        logging.info(f"Extracting keypoints from {len(jobs)} clips with {NUM_WORKERS} workers...")

        X, labels = build_dataset(jobs)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_FILE, X=X, labels=labels)
        logging.info(f"Cached {len(X)} extracted clips in {CACHE_FILE}")

    X, labels = filter_rare_classes(X, labels)
    split_and_save(X, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-cache", action="store_true", help="re-extract keypoints")
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    args = parser.parse_args()

    if args.self_check:
        demo()
    else:
        main(rebuild_cache=args.rebuild_cache)
