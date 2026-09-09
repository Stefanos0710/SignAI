"""Turn the per-word clips cut by `segmentation_videos.py` into a train/val/test dataset.

Input:
    dataset/segments.csv   -- one row per segment: eaf_file, participant, index, ..., transcript
    dataset/word_clips/    -- <eaf_stem>_<participant>_<index:04d>.mp4, cut from those rows

Output:
    dataset/processed/{train,val,test}_data.npz    -- X (N, MAX_FRAMES, 147) float32, y (N,) int64,
                                                        clip_ids (N,) int64, classes (C,) label strings
    dataset/processed/{train,val,test}_images.npz  -- clip_ids (N,) int64 (matches *_data.npz row
                                                        for row, so the two files can be verified
                                                        aligned instead of assumed aligned), plus
                                                        left_hand/right_hand/mouth (N,) object arrays,
                                                        each element a list of MAX_FRAMES JPEG-encoded
                                                        crops (see decode_image_sequence)
    dataset/processed/.keypoint_cache.npz          -- extracted sequences + crops, so a re-split is free

Pipeline (mirrors signai/letter_classification/preprocess_v3.py, but per video instead of per image):
    1) Extract pose + hand keypoints per frame (MediaPipe, same settings as the shared trainer
       preprocessing in signai/preprocessing/train_data.py).
    2) Center on the shoulder midpoint and scale by shoulder distance - using the video average
       shoulders, not per frame, so the signer's own movement stays in the signal.
    3) Interpolate landmarks MediaPipe missed, then Savitzky-Golay smooth over time.
    4) Extract left-hand/right-hand/mouth RGB crops per frame from the same MediaPipe landmarks
       (face mesh mouth points, sliced out of extract_face_keypoints's output), carrying the last
       valid crop forward across frames where that hand/mouth wasn't detected.
    5) Resample / zero-pad every clip (both the keypoint sequence and the three crop sequences,
       using the same frame indices) to MAX_FRAMES frames.
    6) Drop classes with fewer than MIN_SAMPLES_PER_CLASS clips, then split per class (stratified).

Run from the repo root:  python signai/word_classification/preprocessing.py [--rebuild-cache]
"""

import argparse
import csv
import logging
import multiprocessing
import sys
from collections import Counter
from pathlib import Path

import cv2
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
    extract_face_keypoints,
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
NUM_WORKERS = 16
SEED = 42

# --- hand/mouth crop extraction -------------------------------------------------------------
CROP_SIZE = 224          # output crop resolution (square)
CROP_MARGIN = 0.3        # padding added on each side, as a fraction of the bbox size
MIN_CROP_PX = 20         # floor on bbox size (pre-margin), guards degenerate/near-point landmark sets
JPEG_QUALITY = 85
BLACK_CROP = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)

N_FACE_LANDMARKS = len(train_data.FACE_LANDMARKS)  # 93; MOUTH_SLICE below depends on this exact
                                                    # ordering -- see the demo() assertion that
                                                    # catches it if FACE_LANDMARKS ever changes
MOUTH_SLICE = slice(49, 89)  # mouth landmarks' position within extract_face_keypoints()'s
                              # per-frame output (eyebrows 20 + eyes 24 + nose 5 precede it,
                              # mouth is the next 40, cheeks 4 follow)


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


def _resample_indices(n, max_frames=MAX_FRAMES):
    """Shared index-selection for fixing a clip's frame count to `max_frames`.

    Returns the `linspace` indices to keep if `n > max_frames`, otherwise None (meaning:
    keep every frame, pad the rest). Used by both resample_or_pad (keypoints) and
    resample_or_pad_frames (crops) so a keypoint frame and its crop always refer to the
    same source frame.
    """
    if n > max_frames:
        return np.linspace(0, n - 1, max_frames).round().astype(int)
    return None


def resample_or_pad(sequence, max_frames=MAX_FRAMES):
    """Fix the clip to `max_frames`: uniformly subsample if longer, zero-pad at the end if shorter."""
    seq = np.asarray(sequence, dtype=np.float32)
    n = seq.shape[0]
    idx = _resample_indices(n, max_frames)
    if idx is not None:
        return seq[idx]
    if n < max_frames:
        return np.concatenate([seq, np.zeros((max_frames - n, seq.shape[1]), dtype=np.float32)])
    return seq


def resample_or_pad_frames(frames, pad_value=BLACK_CROP, max_frames=MAX_FRAMES):
    """Same fix-up as resample_or_pad, but for a plain list of image arrays (can't be
    stacked into one numpy array before this since frame counts vary clip to clip).
    """
    n = len(frames)
    idx = _resample_indices(n, max_frames)
    if idx is not None:
        return [frames[i] for i in idx]
    if n < max_frames:
        return list(frames) + [pad_value] * (max_frames - n)
    return list(frames)


def bbox_from_points(points_xy, frame_w, frame_h):
    """Pixel bounding box around normalized (x, y) points, padded by CROP_MARGIN and
    clamped to the frame. `points_xy` is an (N, 2+) array-like (extra columns, e.g. z, ignored).
    """
    pts = np.asarray(points_xy, dtype=float)[:, :2]
    x_min, y_min = pts.min(axis=0) * (frame_w, frame_h)
    x_max, y_max = pts.max(axis=0) * (frame_w, frame_h)

    w, h = max(x_max - x_min, MIN_CROP_PX), max(y_max - y_min, MIN_CROP_PX)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_w, half_h = w * (1 + CROP_MARGIN) / 2, h * (1 + CROP_MARGIN) / 2

    x1 = int(np.clip(cx - half_w, 0, frame_w))
    x2 = int(np.clip(cx + half_w, 0, frame_w))
    y1 = int(np.clip(cy - half_h, 0, frame_h))
    y2 = int(np.clip(cy + half_h, 0, frame_h))
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


def crop_and_resize(frame_bgr, bbox):
    """Crop `frame_bgr` to `bbox` (x1, y1, x2, y2) and resize to (CROP_SIZE, CROP_SIZE)."""
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def build_crop_sequence(frames, point_seq):
    """Per-frame CROP_SIZE x CROP_SIZE x 3 crops around `point_seq` (a per-frame list of
    normalized (x, y[, z]) points, or None where nothing was detected that frame).
    A None frame carries the last valid crop forward; frames before the first detection
    are black.
    """
    crops = []
    last_crop = None
    for frame, points in zip(frames, point_seq):
        if points:
            h, w = frame.shape[:2]
            last_crop = crop_and_resize(frame, bbox_from_points(points, w, h))
        crops.append(last_crop if last_crop is not None else BLACK_CROP)
    return crops


def encode_crops(crops):
    """List of uint8 image arrays -> list of JPEG-encoded byte strings."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    return [cv2.imencode(".jpg", crop, encode_params)[1].tobytes() for crop in crops]


def decode_image_sequence(jpeg_bytes_list):
    """Inverse of encode_crops, for consumers: -> (len(jpeg_bytes_list), CROP_SIZE, CROP_SIZE, 3)
    uint8 array in RGB order (storage is BGR, matching this codebase's cv2 convention elsewhere).
    """
    frames = [
        cv2.cvtColor(cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        for b in jpeg_bytes_list
    ]
    return np.stack(frames)


def extract_clip(job):
    """Worker: one clip -> (clip_id, clip_name, label, keypoints, images, failure_reason).

    `keypoints` is a (MAX_FRAMES, N_FEATURES) array, `images` a dict of
    {"left_hand", "right_hand", "mouth"} -> list of MAX_FRAMES JPEG byte strings.
    Both are None together on failure.
    """
    clip_id, clip_path, label = job
    frames = extract_frames(str(clip_path))
    if not frames:
        return clip_id, clip_path.name, label, None, None, "read_failed"

    pose_kp = extract_pose_keypoints(frames)
    hand_kp = extract_hand_keypoints(frames)
    face_kp = extract_face_keypoints(frames)
    avg_left, avg_right = average_shoulders(pose_kp)
    if avg_left is None:
        return clip_id, clip_path.name, label, None, None, "no_pose"

    sequence = []
    left_points, right_points, mouth_points = [], [], []
    for (_, pose), (_, left, right), (_, face) in zip(pose_kp, hand_kp, face_kp):
        pose_arr = np.array(pose, dtype=float) if pose else np.zeros((N_POSE, 3))
        left_arr = np.array(left, dtype=float) if left else np.zeros((N_HAND, 3))
        right_arr = np.array(right, dtype=float) if right else np.zeros((N_HAND, 3))

        all_kp = np.concatenate([pose_arr, left_arr, right_arr], axis=0)
        if pose:
            all_kp = center_keypoints(all_kp, avg_left, avg_right)
            all_kp = normalize_keypoints(all_kp, avg_left, avg_right)
        sequence.append(all_kp)

        left_points.append(left)
        right_points.append(right)
        mouth_points.append(face[MOUTH_SLICE] if face else None)

    sequence = interpolate_missing_keypoints(sequence)
    sequence = apply_temporal_savgol_smoothing(sequence, window_length=9, polyorder=2)
    flat = sequence.reshape(len(sequence), N_FEATURES)

    images = {}
    for name, points in (("left_hand", left_points), ("right_hand", right_points), ("mouth", mouth_points)):
        crops = build_crop_sequence(frames, points)
        crops = resample_or_pad_frames(crops)
        images[name] = encode_crops(crops)

    return clip_id, clip_path.name, label, resample_or_pad(flat), images, None


def init_worker():
    train_data.init_worker()
    logging.getLogger().setLevel(logging.WARNING)  # one INFO line per clip is 9k lines of noise


def build_dataset(jobs, workers=NUM_WORKERS):
    """Extract every clip in parallel.

    Returns clip_ids, X (N, MAX_FRAMES, N_FEATURES), labels, and a dict of
    {"left_hand", "right_hand", "mouth"} -> list of N per-clip JPEG-byte-lists.
    """
    clip_ids, X, labels = [], [], []
    images = {"left_hand": [], "right_hand": [], "mouth": []}
    failed = Counter()

    with multiprocessing.Pool(processes=workers, initializer=init_worker) as pool:
        for clip_id, name, label, features, crops, reason in tqdm(
            pool.imap_unordered(extract_clip, jobs, chunksize=8), total=len(jobs), desc="clips"
        ):
            if features is None:
                failed[reason] += 1
                logging.debug(f"skipping {name}: {reason}")
                continue
            clip_ids.append(clip_id)
            X.append(features)
            labels.append(label)
            for key in images:
                images[key].append(crops[key])

    for reason, count in failed.items():
        logging.info(f"Failed ({reason}): {count} ({count / len(jobs) * 100:.2f}%)")
    return (
        np.array(clip_ids, dtype=np.int64),
        np.array(X, dtype=np.float32),
        np.array(labels),
        {key: np.array(val, dtype=object) for key, val in images.items()},
    )


def filter_rare_classes(clip_ids, X, labels, images, min_samples=MIN_SAMPLES_PER_CLASS):
    counts = Counter(labels)
    keep = np.array([counts[l] >= min_samples for l in labels])
    dropped_classes = sum(1 for c in counts.values() if c < min_samples)
    logging.info(
        f"Dropped {dropped_classes} class(es) with < {min_samples} samples "
        f"({(~keep).sum()} clips); {len(counts) - dropped_classes} classes left"
    )
    return clip_ids[keep], X[keep], labels[keep], {key: val[keep] for key, val in images.items()}


def split_and_save(clip_ids, X, labels, images, output_folder=OUT_DIR, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    """Stratified split -- every class keeps at least one val and one test sample.

    Writes {split}_data.npz (X, y, clip_ids, classes) and {split}_images.npz (clip_ids,
    left_hand, right_hand, mouth), row-aligned by clip_ids in both files.
    """
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
            output_folder / f"{split_name}_data.npz",
            X=X[idx], y=y[idx], clip_ids=clip_ids[idx], classes=classes,
        )
        # images are already JPEG-compressed -- plain savez skips paying zlib twice
        np.savez(
            output_folder / f"{split_name}_images.npz",
            clip_ids=clip_ids[idx],
            **{key: val[idx] for key, val in images.items()},
        )
        logging.info(f"{split_name}: {len(idx)} samples -> {output_folder / f'{split_name}_data.npz'}")

    logging.info(f"Shape per sample: ({MAX_FRAMES}, {N_FEATURES}) | classes: {len(classes)}")


def demo():
    """Self-check: python signai/word_classification/preprocessing.py --self-check"""
    # MOUTH_SLICE assumes extract_face_keypoints() always returns this many landmarks, in this
    # order -- if FACE_LANDMARKS is ever edited upstream, this must fail loudly, not silently
    # crop the wrong face region.
    assert N_FACE_LANDMARKS == len(train_data.FACE_LANDMARKS) == 93, (
        "train_data.FACE_LANDMARKS changed length -- MOUTH_SLICE needs recomputing"
    )

    short = np.ones((5, N_FEATURES), dtype=np.float32)
    padded = resample_or_pad(short)
    assert padded.shape == (MAX_FRAMES, N_FEATURES)
    assert padded[:5].all() and not padded[5:].any(), "padding must be zeros after the real frames"

    long = np.arange(100 * N_FEATURES, dtype=np.float32).reshape(100, N_FEATURES)
    sub = resample_or_pad(long)
    assert sub.shape == (MAX_FRAMES, N_FEATURES)
    assert np.array_equal(sub[0], long[0]) and np.array_equal(sub[-1], long[-1]), "keeps clip ends"

    # resample_or_pad and resample_or_pad_frames must pick identical source frames, so a
    # keypoint frame and its crop stay lined up after resampling.
    dummy_frames = [np.full((CROP_SIZE, CROP_SIZE, 3), i, dtype=np.uint8) for i in range(100)]
    sub_frames = resample_or_pad_frames(dummy_frames)
    assert len(sub_frames) == MAX_FRAMES
    assert np.array_equal(sub_frames[0], dummy_frames[0]) and np.array_equal(sub_frames[-1], dummy_frames[-1])
    kept_idx = _resample_indices(100, MAX_FRAMES)
    assert all(np.array_equal(sub_frames[i], dummy_frames[kept_idx[i]]) for i in range(MAX_FRAMES))

    short_frames = [np.full((CROP_SIZE, CROP_SIZE, 3), 7, dtype=np.uint8) for _ in range(5)]
    padded_frames = resample_or_pad_frames(short_frames)
    assert len(padded_frames) == MAX_FRAMES
    assert all(np.array_equal(f, BLACK_CROP) for f in padded_frames[5:]), "padding must be black frames"

    # bbox_from_points: margin expands, and clamps at the frame edge
    pts = np.array([[0.5, 0.5], [0.6, 0.6]])
    x1, y1, x2, y2 = bbox_from_points(pts, frame_w=100, frame_h=100)
    assert x1 < 50 and x2 > 60, "margin should pad beyond the tight bbox"
    edge_pts = np.array([[0.0, 0.0], [0.02, 0.02]])
    x1, y1, x2, y2 = bbox_from_points(edge_pts, frame_w=100, frame_h=100)
    assert x1 == 0 and y1 == 0, "bbox must clamp to the frame, not go negative"

    crop = crop_and_resize(np.zeros((200, 200, 3), dtype=np.uint8), (10, 10, 100, 100))
    assert crop.shape == (CROP_SIZE, CROP_SIZE, 3) and crop.dtype == np.uint8

    # build_crop_sequence: carries the last valid crop forward, black before the first detection
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    point_seq = [None, None, [(0.4, 0.4, 0.0)] * 3, None]
    crops = build_crop_sequence([frame] * 4, point_seq)
    assert np.array_equal(crops[0], BLACK_CROP) and np.array_equal(crops[1], BLACK_CROP)
    assert np.array_equal(crops[3], crops[2]), "undetected frame after a hit carries it forward"

    # encode/decode round-trip
    encoded = encode_crops([crop_and_resize(frame, (0, 0, 50, 50))])
    decoded = decode_image_sequence(encoded)
    assert decoded.shape == (1, CROP_SIZE, CROP_SIZE, 3) and decoded.dtype == np.uint8

    labels = np.array(["A"] * 20 + ["B"] * 5 + ["C"] * 2)
    clip_ids = np.arange(len(labels))
    X = np.zeros((len(labels), MAX_FRAMES, N_FEATURES), dtype=np.float32)
    dummy_images = {k: np.array([[b"x"] * MAX_FRAMES] * len(labels), dtype=object) for k in
                     ("left_hand", "right_hand", "mouth")}
    ids_kept, X_kept, labels_kept, images_kept = filter_rare_classes(clip_ids, X, labels, dummy_images)
    assert set(labels_kept.tolist()) == {"A", "B"} and len(X_kept) == 25
    assert len(ids_kept) == 25 and all(len(v) == 25 for v in images_kept.values())

    names = load_labels()
    existing = [p.name for p in CLIPS_DIR.glob("*.mp4")][:50] if CLIPS_DIR.exists() else []
    for name in existing:
        assert name in names, f"clip {name} has no row in {SEGMENTS_CSV.name}"
    print(f"self-check passed ({len(existing)} real clip name(s) checked)")


def main(rebuild_cache=False):
    if CACHE_FILE.exists() and not rebuild_cache:
        logging.info(f"Loading cached keypoints from {CACHE_FILE} (--rebuild-cache to re-extract)")
        cached = np.load(CACHE_FILE, allow_pickle=True)
        clip_ids, X, labels = cached["clip_ids"], cached["X"], cached["labels"]
        images = {key: cached[key] for key in ("left_hand", "right_hand", "mouth")}
    else:
        names = load_labels()
        clips = [p for p in sorted(CLIPS_DIR.glob("*.mp4")) if p.name in names]
        jobs = [(clip_id, p, names[p.name]) for clip_id, p in enumerate(clips)]
        missing = len(list(CLIPS_DIR.glob("*.mp4"))) - len(jobs)
        if missing:
            logging.warning(f"{missing} clip(s) without a segments.csv row -- skipped")
        logging.info(f"Extracting keypoints from {len(jobs)} clips with {NUM_WORKERS} workers...")

        clip_ids, X, labels, images = build_dataset(jobs)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        # images are already JPEG-compressed -- plain savez skips paying zlib twice
        np.savez(CACHE_FILE, clip_ids=clip_ids, X=X, labels=labels, **images)
        logging.info(f"Cached {len(X)} extracted clips in {CACHE_FILE}")

    clip_ids, X, labels, images = filter_rare_classes(clip_ids, X, labels, images)
    split_and_save(clip_ids, X, labels, images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-cache", action="store_true", help="re-extract keypoints")
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    args = parser.parse_args()

    if args.self_check:
        demo()
    else:
        main(rebuild_cache=args.rebuild_cache)
