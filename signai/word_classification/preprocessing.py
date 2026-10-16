"""Turn the per-word clips cut by `segmentation_videos.py` into a train/val/test dataset.

Input:
    dataset/segments.csv   -- one row per segment: eaf_file, participant, index, ..., transcript
    dataset/word_clips/    -- <eaf_stem>_<participant>_<index:04d>.mp4, cut from those rows

Output:
    dataset/processed/{train,val,test}_data.npz    -- X (N, MAX_FRAMES, N_FEATURES) float32 body-normalized
                                                        keypoints (49 landmarks x (x, y, z), flattened per
                                                        frame -- the model renders the per-landmark heatmaps
                                                        on the GPU, see models/heatmap.py), y (N,) int64,
                                                        clip_ids (N,) int64, classes (C,) label strings
    dataset/processed/{train,val,test}_images.npz  -- clip_ids (N,) int64 (matches *_data.npz row
                                                        for row, so the two files can be verified
                                                        aligned instead of assumed aligned), plus
                                                        left_hand/right_hand/mouth (N, MAX_FRAMES) object
                                                        arrays of JPEG-encoded crops (see
                                                        decode_image_sequence)
    dataset/processed/shards/shard_*.npz           -- checkpoints: every SHARD_SIZE extracted clips are saved
                                                        as they finish, so an interrupted run resumes where it
                                                        stopped and a re-split is free
    dataset/processed/progress.json                -- done/total/percent/rate/ETA, rewritten every checkpoint
    dataset/processed/preprocessing.log            -- the console's log lines, for `tail -f` outside tmux
    dataset/processed/worker_stderr.log            -- MediaPipe's native log output from the workers

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
    6) Applying Shades-of-Gray Coloring to the crops to reduce skin tone bias
    7) Drop classes with fewer than MIN_SAMPLES_PER_CLASS clips, then split per class (stratified).

Run from the repo root:
    python signai/word_classification/preprocessing.py                  # extract (resumes) + split
    python signai/word_classification/preprocessing_dashboard.py        # live progress/ETA dashboard
    python signai/word_classification/preprocessing.py --status         # one dashboard snapshot
    python signai/word_classification/preprocessing.py --limit 1000     # quick test on 1000 clips
    python signai/word_classification/preprocessing.py --rebuild-cache  # drop shards/, start over
"""

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta
from pathlib import Path

# read by MediaPipe/TensorFlow's C++ logging when they're imported (via train_data below)
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402
from tqdm.contrib.logging import logging_redirect_tqdm  # noqa: E402

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
from signai.word_classification.paths import DATASET_DIR, PROCESSED_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

CLIPS_DIR = DATASET_DIR / "word_clips"
SEGMENTS_CSV = DATASET_DIR / "segments.csv"
OUT_DIR = PROCESSED_DIR
SHARD_DIR = OUT_DIR / "shards"
PROGRESS_FILE = OUT_DIR / "progress.json"
LOG_FILE = OUT_DIR / "preprocessing.log"
WORKER_LOG_FILE = OUT_DIR / "worker_stderr.log"

N_POSE = len(POSE_LANDMARKS)
N_HAND = 21
N_LANDMARKS = N_POSE + 2 * N_HAND          # 7 pose + 42 hand = 49
N_FEATURES = N_LANDMARKS * 3               # 147 per frame (no face -- words are hand/arm shapes)
MAX_FRAMES = 32                            # ~99th percentile of the segment durations at 50 fps
MIN_SAMPLES_PER_CLASS = 5
NUM_WORKERS = os.cpu_count()               # a MediaPipe worker keeps ~1.2 cores busy
SHARD_SIZE = 2000                          # clips per checkpoint file
PROGRESS_INTERVAL = 30                     # seconds between progress.json updates between checkpoints
MAX_POOL_RESTARTS = 3
SEED = 42

# --- hand/mouth crop extraction -------------------------------------------------------------
CROP_SIZE = 112          # output crop resolution (square). Hands are ~60-75 px and mouths ~25-30 px
                         # in the source video, so a bigger crop only stores upscaled pixels; the
                         # DINOv3 processor resizes to its own 224 input either way
CROP_MARGIN = 0.3        # padding added on each side, as a fraction of the bbox size
MIN_CROP_PX = 20         # floor on bbox size (pre-margin), guards degenerate/near-point landmark sets
JPEG_QUALITY = 85
CROP_TYPES = ("left_hand", "right_hand", "mouth")
BLACK_CROP = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
# one shared bytes object for every black crop, so pickle stores it once per file
BLACK_JPEG = cv2.imencode(".jpg", BLACK_CROP, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()

N_FACE_LANDMARKS = len(train_data.FACE_LANDMARKS)  # 93; MOUTH_SLICE below depends on this exact
                                                    # ordering -- see the demo() assertion that
                                                    # catches it if FACE_LANDMARKS ever changes
MOUTH_SLICE = slice(49, 89)  # mouth landmarks' position within extract_face_keypoints()'s
                              # per-frame output (eyebrows 20 + eyes 24 + nose 5 precede it,
                              # mouth is the next 40, cheeks 4 follow)
MINKOWSKI_EXPONENT = 6 #  # for Shades-of-Gray color constancy (see apply_shades_of_gray() below)


def fix_missing_hand_detections(sequence, left_detected, right_detected):
    """(T, N_LANDMARKS, 3) -> same shape. A hand MediaPipe never detected in a
    frame is filled with raw (0, 0, 0) (see extract_clip); once center_keypoints/
    normalize_keypoints have run, that's no longer exactly (0, 0, 0), it's shifted
    to some fixed off-body point instead -- so interpolate_missing_keypoints's
    exact-zero check (which runs after normalization) doesn't catch it, and a
    hand can visibly teleport in from that fixed point once real tracking picks
    up. left_detected/right_detected are the true per-frame detection flags
    (from before normalization), so this catches what that check misses.

    Linearly interpolates each hand's landmarks across its undetected frames
    from the nearest frames where it actually was detected.
    """
    sequence = np.asarray(sequence, dtype=np.float64).copy()
    frame_idx = np.arange(sequence.shape[0])
    for detected, lo, hi in (
        (left_detected, N_POSE, N_POSE + N_HAND),
        (right_detected, N_POSE + N_HAND, N_POSE + 2 * N_HAND),
    ):
        detected = np.asarray(detected, dtype=bool)
        valid = np.where(detected)[0]
        if len(valid) == 0 or len(valid) == len(detected):
            continue  # never detected at all, or always detected -- nothing to fix
        for j in range(lo, hi):
            for k in range(3):
                sequence[:, j, k] = np.interp(frame_idx, valid, sequence[valid, j, k])
    return sequence


def anchor_hands_to_wrist(sequence):
    """(T, N_LANDMARKS, 3) -> same shape. Pose and hands are two independently
    run MediaPipe models (pose.process() / hands.process()) that don't always
    agree on where the wrist is -- confirmed on real data the gap between the
    pose model's wrist (POSE_LANDMARKS index 5/6) and the hand model's own
    wrist (landmark 0 of each 21-point hand block) is often bigger than a
    whole forearm. Left as-is, a hand keeps its correct shape but appears to
    float away from the arm it's attached to.

    Rigidly translates each hand's 21 landmarks so its own landmark 0 sits
    exactly on the corresponding pose wrist -- keeps the hand model's finger
    articulation, anchors its position to the much more stable pose skeleton.
    """
    sequence = np.asarray(sequence, dtype=np.float64).copy()
    left_lo, left_hi = N_POSE, N_POSE + N_HAND
    right_lo, right_hi = N_POSE + N_HAND, N_POSE + 2 * N_HAND
    left_offset = sequence[:, 5, :] - sequence[:, left_lo, :]    # pose left wrist - hand's own wrist
    right_offset = sequence[:, 6, :] - sequence[:, right_lo, :]  # pose right wrist - hand's own wrist
    sequence[:, left_lo:left_hi, :] += left_offset[:, None, :]
    sequence[:, right_lo:right_hi, :] += right_offset[:, None, :]
    return sequence


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
    A None frame carries the last valid crop forward (the same array object); frames
    before the first detection are BLACK_CROP.
    """
    crops = []
    last_crop = None
    for frame, points in zip(frames, point_seq):
        if points:
            h, w = frame.shape[:2]
            last_crop = crop_and_resize(frame, bbox_from_points(points, w, h))
        crops.append(last_crop if last_crop is not None else BLACK_CROP)
    return crops


def encode_crop(crop):
    """uint8 image array -> JPEG-encoded byte string."""
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()


def decode_image_sequence(jpeg_bytes_list):
    """Inverse of encode_crop, for consumers: -> (len(jpeg_bytes_list), CROP_SIZE, CROP_SIZE, 3)
    uint8 array in RGB order (storage is BGR, matching this codebase's cv2 convention elsewhere).
    """
    frames = [
        cv2.cvtColor(cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        for b in jpeg_bytes_list
    ]
    return np.stack(frames)


def apply_shades_of_gray(crop, p=MINKOWSKI_EXPONENT):
    """Shades-of-Gray color constancy: per-channel illuminant estimate via the p-th
    Minkowski norm (p=1 is plain gray-world; higher p weights bright pixels more), then
    rescale channels so that estimate is neutral. Cuts skin-tone-driven color differences
    between signers before crops reach the model. A black crop passes through unchanged.
    """
    crop_f64 = crop.astype(np.float64)
    illum = np.power(np.mean(np.power(crop_f64, p), axis=(0, 1)), 1.0 / p)
    if not illum.any():
        return crop
    scale = illum.mean() / illum
    return np.clip(crop_f64 * scale, 0, 255).astype(np.uint8)


def build_image_sequence(frames, point_seq):
    """Crops -> MAX_FRAMES -> Shades-of-Gray -> JPEG, one list of MAX_FRAMES byte strings.

    Resampling first means frames that get dropped are never color-corrected or encoded, and a
    crop repeated by carry-forward or padding is only processed once -- the repeats share one
    bytes object, which pickle (np.savez of an object array) then stores once.
    """
    crops = resample_or_pad_frames(build_crop_sequence(frames, point_seq))
    encoded = {id(BLACK_CROP): BLACK_JPEG}
    for crop in crops:
        if id(crop) not in encoded:
            encoded[id(crop)] = encode_crop(apply_shades_of_gray(crop))
    return [encoded[id(crop)] for crop in crops]


def extract_clip(job):
    """Worker: one (clip_path, label) -> (clip_name, label, keypoints, images, failure_reason).

    `keypoints` is a (MAX_FRAMES, N_FEATURES) float32 array, `images` a dict of
    CROP_TYPES -> list of MAX_FRAMES JPEG byte strings. Both are None together on failure.
    """
    clip_path, label = job
    frames = extract_frames(str(clip_path))
    if not frames:
        return clip_path.name, label, None, None, "read_failed"

    pose_kp = extract_pose_keypoints(frames)
    hand_kp = extract_hand_keypoints(frames)
    face_kp = extract_face_keypoints(frames)
    avg_left, avg_right = average_shoulders(pose_kp)
    if avg_left is None:
        return clip_path.name, label, None, None, "no_pose"

    sequence = []
    left_detected, right_detected = [], []
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

        left_detected.append(left is not None)
        right_detected.append(right is not None)
        left_points.append(left)
        right_points.append(right)
        mouth_points.append(face[MOUTH_SLICE] if face else None)

    sequence = interpolate_missing_keypoints(sequence)
    sequence = fix_missing_hand_detections(sequence, left_detected, right_detected)
    sequence = anchor_hands_to_wrist(sequence)
    sequence = apply_temporal_savgol_smoothing(sequence, window_length=9, polyorder=2)
    flat = sequence.reshape(len(sequence), N_FEATURES)
    keypoints = resample_or_pad(flat)  # (MAX_FRAMES, N_FEATURES), same frame selection as the crops

    images = {
        name: build_image_sequence(frames, points)
        for name, points in zip(CROP_TYPES, (left_points, right_points, mouth_points))
    }
    return clip_path.name, label, keypoints, images, None


def init_worker(stderr_path=None):
    if stderr_path is not None:
        # MediaPipe/TFLite print their init chatter straight to fd 2, which buries the progress
        # bar; Python exceptions still reach the parent through the pool
        fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 2)
        os.close(fd)
    train_data.init_worker()
    logging.getLogger().setLevel(logging.WARNING)  # one INFO line per clip is pure noise


def ensure_mediapipe_models():
    """Pose() downloads its model on first use, non-atomically, so workers starting together
    race on the download and can leave a truncated .tflite that segfaults every later Pose().
    Fetching it here, before any worker exists, avoids that race."""
    from mediapipe.python.solutions.pose import _download_oss_pose_landmark_model
    _download_oss_pose_landmark_model(0)  # model_complexity=0, as in train_data.init_worker


def format_duration(seconds):
    seconds = int(seconds)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    if days:
        return f"{days}d {hours}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


class ProgressReporter:
    """Rewrites PROGRESS_FILE, and logs a progress line unless log=False. The rate (and so the
    ETA) only counts this run's clips, so time spent before a resume doesn't skew it."""

    def __init__(self, total, done, failed):
        self.total = total
        self.done = done
        self.failed = failed
        self.start_done = done
        self.started = time.time()
        self.started_at = datetime.now()
        self.last_write = 0.0

    def write(self, phase="extracting", log=True):
        self.last_write = time.time()
        now = datetime.now()
        elapsed = time.time() - self.started
        rate = (self.done - self.start_done) / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.done
        eta = remaining / rate if rate > 0 and remaining > 0 else None
        finish = now + timedelta(seconds=eta) if eta is not None else None
        status = {
            "phase": phase,
            "done": self.done,
            "total": self.total,
            "percent": round(100 * self.done / self.total, 2),
            "failed": self.failed,
            "clips_per_sec": round(rate, 2),
            "eta": format_duration(eta) if eta is not None else None,
            "eta_finish": finish.isoformat(timespec="minutes") if finish else None,
            "started": self.started_at.isoformat(timespec="seconds"),
            "updated": now.isoformat(timespec="seconds"),
            "pid": os.getpid(),
        }
        tmp = PROGRESS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, PROGRESS_FILE)

        if not log:
            return
        if phase != "extracting":
            logging.info(f"phase: {phase}")
        elif eta is not None:
            logging.info(
                f"checkpoint: {self.done:,}/{self.total:,} clips ({status['percent']:.1f}%), "
                f"{self.failed:,} failed | {rate:.1f} clips/s | ETA {status['eta']} "
                f"(~{finish:%a %d.%m. %H:%M})"
            )
        else:
            logging.info(f"checkpoint: {self.done:,}/{self.total:,} clips ({status['percent']:.1f}%)")


def load_done_names():
    """-> (names extracted OK, names that failed) across all checkpoint shards."""
    ok, failed = set(), set()
    for shard in SHARD_DIR.glob("shard_*.npz"):
        with np.load(shard) as f:
            ok.update(f["names"].tolist())
            failed.update(f["failed_names"].tolist())
    return ok, failed


def save_shard(rows, failed):
    """rows: [(name, label, keypoints, images)], failed: [(name, reason)] -> next shard file,
    written to a temp name first so a kill mid-write never leaves a half-written shard."""
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    index = max((int(p.stem.split("_")[1]) for p in SHARD_DIR.glob("shard_*.npz")), default=-1) + 1
    path = SHARD_DIR / f"shard_{index:05d}.npz"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(
            f,
            names=np.array([r[0] for r in rows], dtype=str),
            labels=np.array([r[1] for r in rows], dtype=str),
            X=np.array([r[2] for r in rows], dtype=np.float32).reshape(len(rows), MAX_FRAMES, N_FEATURES),
            failed_names=np.array([name for name, _ in failed], dtype=str),
            failed_reasons=np.array([reason for _, reason in failed], dtype=str),
            # images are already JPEG-compressed -- plain savez skips paying zlib twice
            **{
                key: np.array([r[3][key] for r in rows], dtype=object).reshape(len(rows), MAX_FRAMES)
                for key in CROP_TYPES
            },
        )
    os.replace(tmp, path)


def extract_pending(jobs, workers, reporter):
    """Extract every clip in `jobs` that isn't in a shard yet, checkpointing every SHARD_SIZE.

    ProcessPoolExecutor rather than multiprocessing.Pool: when a worker dies (segfault, OOM
    kill), Pool silently starts a replacement and waits forever for the lost task, while this
    raises BrokenProcessPool.
    """
    ok, failed_before = load_done_names()
    pending = [job for job in jobs if job[0].name not in ok and job[0].name not in failed_before]
    if not pending:
        return
    logging.info(
        f"{len(jobs) - len(pending):,} clips already checkpointed, extracting {len(pending):,} "
        f"with {workers} workers"
    )

    rows, failed = [], []

    def checkpoint():
        save_shard(rows, failed)
        rows.clear()
        failed.clear()
        reporter.write()

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(str(WORKER_LOG_FILE),)
    )
    try:
        with tqdm(total=reporter.total, initial=reporter.done, desc="clips", unit="clip",
                  smoothing=0.05, dynamic_ncols=True) as bar:
            for name, label, keypoints, images, reason in pool.map(extract_clip, pending, chunksize=4):
                if reason is None:
                    rows.append((name, label, keypoints, images))
                else:
                    failed.append((name, reason))
                    reporter.failed += 1
                reporter.done += 1
                bar.update(1)
                if len(rows) + len(failed) >= SHARD_SIZE:
                    checkpoint()
                elif time.time() - reporter.last_write >= PROGRESS_INTERVAL:
                    reporter.write(log=False)
    finally:
        if rows or failed:
            checkpoint()
        pool.shutdown(wait=False, cancel_futures=True)


def extract_all(jobs, workers, reporter):
    for attempt in range(MAX_POOL_RESTARTS + 1):
        try:
            extract_pending(jobs, workers, reporter)
            return
        except BrokenProcessPool:
            if attempt == MAX_POOL_RESTARTS:
                raise RuntimeError(
                    f"worker processes crashed {attempt + 1} times (see {WORKER_LOG_FILE}). Everything "
                    f"finished so far is saved in {SHARD_DIR}, rerun to resume. If workers crash before "
                    "finishing any clip, a MediaPipe model file is likely corrupt (e.g. a truncated "
                    "mediapipe/modules/pose_landmark/pose_landmark_lite.tflite -- delete it to re-download)."
                ) from None
            logging.error(
                f"a worker process crashed -- finished clips are saved, restarting the pool "
                f"({attempt + 1}/{MAX_POOL_RESTARTS})"
            )


def load_shards(jobs):
    """Merge the checkpoint shards into one dataset, keeping only clips that are in `jobs` (so a
    --limit run or a changed segments.csv never pulls in stale clips). clip_id is the clip's
    position in `jobs`, which is sorted by filename, so it's stable across runs.
    """
    index = {path.name: i for i, (path, _) in enumerate(jobs)}
    labels_by_name = {path.name: label for path, label in jobs}
    clip_ids, X, labels = [], [], []
    images = {key: [] for key in CROP_TYPES}
    failed = Counter()
    for shard in tqdm(sorted(SHARD_DIR.glob("shard_*.npz")), desc="merging shards", unit="shard"):
        with np.load(shard, allow_pickle=True) as f:
            names = f["names"]
            keep = np.array([name in index for name in names], dtype=bool)
            clip_ids.append(np.array([index[name] for name in names[keep]], dtype=np.int64))
            labels.append(np.array([labels_by_name[name] for name in names[keep]], dtype=str))
            X.append(f["X"][keep])
            for key in CROP_TYPES:
                images[key].append(f[key][keep])
            failed.update(r for name, r in zip(f["failed_names"], f["failed_reasons"]) if name in index)

    for reason, count in failed.items():
        logging.info(f"Failed ({reason}): {count} ({count / len(jobs) * 100:.2f}%)")
    return (
        np.concatenate(clip_ids),
        np.concatenate(X),
        np.concatenate(labels),
        {key: np.concatenate(val) for key, val in images.items()},
    )


def filter_rare_classes(clip_ids, X, labels, images, min_samples=MIN_SAMPLES_PER_CLASS):
    if len(labels) == 0:
        raise RuntimeError(
            f"No usable word clips found in {CLIPS_DIR}. "
            "Download the videos, run segmentation_videos.py, then rerun with --rebuild-cache."
        )

    counts = Counter(labels)
    keep = np.array([counts[l] >= min_samples for l in labels], dtype=bool)
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

    # fix_missing_hand_detections: undetected-hand frames get interpolated from real ones,
    # detected frames are untouched
    seq = np.zeros((5, N_LANDMARKS, 3))
    seq[:, N_POSE, :] = [[9, 9, 9], [1, 1, 1], [9, 9, 9], [9, 9, 9], [3, 3, 3]]  # left hand landmark 0
    left_detected = [False, True, False, False, True]
    right_detected = [True] * 5
    fixed = fix_missing_hand_detections(seq, left_detected, right_detected)
    assert np.allclose(fixed[1, N_POSE], [1, 1, 1]) and np.allclose(fixed[4, N_POSE], [3, 3, 3])
    assert np.allclose(fixed[2, N_POSE], [1 + 2 / 3, 1 + 2 / 3, 1 + 2 / 3]), (
        "interpolates linearly between the two real frames (frame 1 -> frame 4)"
    )

    # anchor_hands_to_wrist: each hand's own landmark 0 ends up exactly on the pose wrist,
    # rest of the hand keeps its shape (same offset from landmark 0)
    seq = np.zeros((1, N_LANDMARKS, 3))
    seq[0, 5] = [1.0, 2.0, 0.0]   # pose left wrist
    seq[0, 6] = [-1.0, 2.0, 0.0]  # pose right wrist
    seq[0, N_POSE] = [0.0, 0.0, 0.0]              # left hand's own wrist, elsewhere
    seq[0, N_POSE + 1] = [0.1, 0.1, 0.0]           # another left-hand landmark
    anchored = anchor_hands_to_wrist(seq)
    assert np.allclose(anchored[0, N_POSE], seq[0, 5]), "hand wrist must land on pose wrist"
    assert np.allclose(anchored[0, N_POSE + 1] - anchored[0, N_POSE], [0.1, 0.1, 0.0]), "shape preserved"

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

    # apply_shades_of_gray: black padding crops pass through untouched, colored crops get
    # their channel means pulled together
    assert np.array_equal(apply_shades_of_gray(BLACK_CROP), BLACK_CROP)
    tinted = np.zeros((4, 4, 3), dtype=np.uint8)
    tinted[..., 0] = 200  # strong red cast
    tinted[..., 1] = 50
    tinted[..., 2] = 50
    corrected = apply_shades_of_gray(tinted)
    means = corrected.astype(np.float64).mean(axis=(0, 1))
    assert means.max() - means.min() < tinted.astype(np.float64).mean(axis=(0, 1)).ptp(), (
        "shades-of-gray should narrow the channel-mean spread, not widen it"
    )

    # build_image_sequence: always MAX_FRAMES JPEGs, repeated crops share one bytes object
    jpegs = build_image_sequence([frame] * 4, point_seq)
    assert len(jpegs) == MAX_FRAMES
    assert jpegs[0] is BLACK_JPEG and jpegs[-1] is BLACK_JPEG, "leading black + padding share BLACK_JPEG"
    assert jpegs[3] is jpegs[2], "carried-forward crop is encoded once"
    assert decode_image_sequence(jpegs).shape == (MAX_FRAMES, CROP_SIZE, CROP_SIZE, 3)

    # encode/decode round-trip
    decoded = decode_image_sequence([encode_crop(crop_and_resize(frame, (0, 0, 50, 50)))])
    assert decoded.shape == (1, CROP_SIZE, CROP_SIZE, 3) and decoded.dtype == np.uint8

    assert format_duration(59) == "0m 59s" and format_duration(3 * 3600 + 120) == "3h 02m"
    assert format_duration(2 * 86400 + 3600) == "2d 1h 00m"

    labels = np.array(["A"] * 20 + ["B"] * 5 + ["C"] * 2)
    clip_ids = np.arange(len(labels))
    X = np.zeros((len(labels), MAX_FRAMES, N_FEATURES), dtype=np.float32)
    dummy_images = {k: np.array([[b"x"] * MAX_FRAMES] * len(labels), dtype=object) for k in CROP_TYPES}
    ids_kept, X_kept, labels_kept, images_kept = filter_rare_classes(clip_ids, X, labels, dummy_images)
    assert set(labels_kept.tolist()) == {"A", "B"} and len(X_kept) == 25
    assert len(ids_kept) == 25 and all(len(v) == 25 for v in images_kept.values())

    names = load_labels()
    existing = [p.name for p in CLIPS_DIR.glob("*.mp4")][:50] if CLIPS_DIR.exists() else []
    for name in existing:
        assert name in names, f"clip {name} has no row in {SEGMENTS_CSV.name}"
    print(f"self-check passed ({len(existing)} real clip name(s) checked)")


def list_jobs():
    """-> [(clip_path, label)] sorted by filename, for every clip with a segments.csv row."""
    names = load_labels()
    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    jobs = [(p, names[p.name]) for p in clips if p.name in names]
    if len(clips) > len(jobs):
        logging.warning(f"{len(clips) - len(jobs)} clip(s) without a segments.csv row -- skipped")
    return jobs


def main(rebuild_cache=False, workers=NUM_WORKERS, limit=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().addHandler(file_handler)

    if rebuild_cache and SHARD_DIR.exists():
        logging.info(f"--rebuild-cache: removing {SHARD_DIR}")
        shutil.rmtree(SHARD_DIR)

    jobs = list_jobs()[:limit]
    if not jobs:
        raise RuntimeError(f"No word clips found in {CLIPS_DIR} -- run segmentation_videos.py first.")

    ok, failed = load_done_names()
    reporter = ProgressReporter(
        total=len(jobs),
        done=sum(p.name in ok or p.name in failed for p, _ in jobs),
        failed=sum(p.name in failed for p, _ in jobs),
    )
    reporter.write()

    ensure_mediapipe_models()
    with logging_redirect_tqdm():
        extract_all(jobs, workers, reporter)

    reporter.write(phase="merging + splitting")
    clip_ids, X, labels, images = load_shards(jobs)
    clip_ids, X, labels, images = filter_rare_classes(clip_ids, X, labels, images)
    split_and_save(clip_ids, X, labels, images)
    reporter.write(phase="done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild-cache", action="store_true", help="delete the checkpoint shards and re-extract")
    parser.add_argument("--status", action="store_true", help="print a snapshot of the progress dashboard")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help=f"worker processes (default {NUM_WORKERS})")
    parser.add_argument("--limit", type=int, help="only use the first N clips, for a quick end-to-end test")
    parser.add_argument("--self-check", action="store_true", help="run the self-check and exit")
    args = parser.parse_args()

    if args.self_check:
        demo()
    elif args.status:
        from signai.word_classification.preprocessing_dashboard import show
        show(once=True)
    else:
        main(rebuild_cache=args.rebuild_cache, workers=args.workers, limit=args.limit)
