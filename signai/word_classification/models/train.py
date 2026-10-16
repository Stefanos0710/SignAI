"""
1) Loading the dataset:
       - load_split(split: str)      -- read the npz files, keep crop JPEGs undecoded
       - create_dataset(split: str)  -- wrap load_split into a tf.data.Dataset that decodes
                                         each clip's crops lazily, per batch
2) Training:
       - build_dataset(split, ...)   -- pose heatmaps + cached hand/face DINOv3 features -> y,
                                         as a tf.data.Dataset (WordClassifier's input, not the
                                         raw crops create_dataset() decodes)
       - train_main(...)             -- compile WordClassifier, fit, save checkpoints, resume
                                         automatically if interrupted (see RESUME below)

Run from the repo root:
    python signai/word_classification/models/train.py                 # train (resumes if interrupted)
    python signai/word_classification/models/training_dashboard.py    # live progress/ETA dashboard
    python signai/word_classification/models/train.py --status        # one dashboard snapshot

RESUME: every epoch writes checkpoints/word_classifier_latest.keras + state.json (completed
epochs, best val_accuracy so far). Rerunning the same command picks up from there -- Ctrl-C or a
crash loses at most the epoch in progress, not the whole run. --restart ignores that state and
starts a fresh run (the previous checkpoints are overwritten once the new run reaches their epoch).
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.models.gpu_env import ensure_gpu_env  # noqa: E402
ensure_gpu_env()  # must run before `import tensorflow` -- see gpu_env.py

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402

from signai.word_classification.models.model import WordClassifier  # noqa: E402
from signai.word_classification.models.hand_stream import load_hand_features  # noqa: E402
from signai.word_classification.models.face_stream import load_face_features  # noqa: E402
from signai.word_classification.models.progress import format_duration  # noqa: E402
from signai.word_classification.paths import PROCESSED_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

DATASET_DIR = str(PROCESSED_DIR)
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
LATEST_CKPT = CHECKPOINT_DIR / "word_classifier_latest.keras"
BEST_CKPT = CHECKPOINT_DIR / "word_classifier_best.keras"
STATE_FILE = CHECKPOINT_DIR / "state.json"
PROGRESS_FILE = CHECKPOINT_DIR / "progress.json"
LOG_FILE = CHECKPOINT_DIR / "train.log"
PROGRESS_INTERVAL = 5  # seconds between progress.json updates within an epoch

SPLITS = ["train", "val", "test"]
CROP_TYPES = ("left_hand", "right_hand", "mouth")

# training parameters
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-4


def load_split(split: str, with_crops: bool = True):
    """Load a dataset split (train/val/test) from dataset/processed/.

    Crops are returned as the raw JPEG byte strings stored in {split}_images.npz, not
    decoded to pixels.
    create_dataset() decodes lazily, one batch at a time, via tf.io.decode_jpeg.

    with_crops=False skips reading the (large, object-array) crop columns from
    {split}_images.npz entirely -- np.load's NpzFile only unpickles an array when it's
    indexed, so build_dataset (which doesn't need the raw crops, only the cached DINOv3
    features) doesn't pay to decompress e.g. train's 21 GB of JPEG bytes just to discard them.

    Returns:
        X: body-normalized keypoints, (N, 32, N_FEATURES) float32 -- the model renders the
            heatmaps itself (see model.py)
        y: label ids, (N,) int64
        classes: label names, (num_classes,) str, classes[y[i]] == label of sample i
        crops: dict with "left_hand"/"right_hand"/"mouth", each (N, 32) object array of
            JPEG-encoded bytes, or None if with_crops=False
    """
    data = np.load(os.path.join(DATASET_DIR, f"{split}_data.npz"), allow_pickle=True)
    images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npz"), allow_pickle=True)

    # the two files are only row-aligned by construction (preprocessing.py writes clip_ids
    # to both)
    assert np.array_equal(data["clip_ids"], images["clip_ids"]), (
        f"{split}: clip_ids mismatch between _data.npz and _images.npz"
    )

    X, y, classes = data["X"], data["y"], data["classes"]
    crops = {crop_type: images[crop_type] for crop_type in CROP_TYPES} if with_crops else None

    return X, y, classes, crops


def create_dataset(split: str, batch_size: int = 8, shuffle: bool = False):
    """Build a tf.data.Dataset for a split, decoding each clip's crop JPEGs lazily
    (tf.io.decode_jpeg, inside the pipeline)

    Returns:
        dataset: yields ((X, crops), y) batches, X is (batch, 32, N_FEATURES) float32, crops is a
            dict "left_hand"/"right_hand"/"mouth" -> (batch, 32, CROP_SIZE, CROP_SIZE, 3)
            uint8 RGB, y is (batch,) int64 label ids.
        classes: label names, (num_classes,) str, classes[y[i]] == label of sample i
    """
    X, y, classes, crops = load_split(split)
    n_frames = X.shape[1]

    def gen():
        for i in range(len(y)):
            yield X[i], {crop_type: crops[crop_type][i] for crop_type in CROP_TYPES}, y[i]

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32),
            {crop_type: tf.TensorSpec(shape=(n_frames,), dtype=tf.string) for crop_type in CROP_TYPES},
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )

    def decode(x, jpeg_crops, label):
        decoded = {
            crop_type: tf.map_fn(
                lambda b: tf.io.decode_jpeg(b, channels=3),
                jpeg_crops[crop_type],
                fn_output_signature=tf.uint8,
            )
            for crop_type in CROP_TYPES
        }
        return (x, decoded), label

    if shuffle:
        dataset = dataset.shuffle(len(y))

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, classes


def build_dataset(split: str, batch_size: int = BATCH_SIZE, shuffle: bool = False):
    """WordClassifier's actual input for a split: pose heatmaps (load_split) +
    cached hand/face DINOv3 features (load_hand_features/load_face_features) -> y.
    Everything here is already a small float array (unlike the raw JPEG crops
    create_dataset() decodes), so from_tensor_slices is enough.
    """
    X, y, classes, _crops = load_split(split, with_crops=False)
    _, left_hand, right_hand = load_hand_features(split)
    _, mouth = load_face_features(split)

    dataset = tf.data.Dataset.from_tensor_slices(((X, left_hand, right_hand, mouth), y))
    if shuffle:
        dataset = dataset.shuffle(len(y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, len(y), classes


class TrainingProgress(keras.callbacks.Callback):
    """Writes CHECKPOINT_DIR/progress.json (for training_dashboard.py) and
    CHECKPOINT_DIR/state.json (epoch/best-metric, for resuming) after every epoch, plus a
    lighter-weight progress.json update every PROGRESS_INTERVAL seconds *within* an epoch so
    the dashboard has a live step/loss/accuracy/ETA during a long epoch, not just at its end.
    """

    def __init__(self, total_epochs, steps_per_epoch, initial_best_val_accuracy=None, initial_best_epoch=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.best_val_accuracy = initial_best_val_accuracy
        self.best_epoch = initial_best_epoch
        self.started_at = datetime.now()
        self.started = time.time()
        self.epoch_history = []  # (epoch_seconds,) per completed epoch this run, for the dashboard
        self._last_write = 0.0
        self._epoch_start = None
        self._epoch = 0
        self._step = 0
        self._logs = {}

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch + 1  # 1-indexed for humans
        self._epoch_start = time.time()
        self._step = 0
        self._logs = {}
        self._write(phase="training")

    def on_train_batch_end(self, batch, logs=None):
        self._step = batch + 1
        self._logs = logs or {}
        if time.time() - self._last_write >= PROGRESS_INTERVAL:
            self._write(phase="training", log=False)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._logs = logs
        epoch_seconds = time.time() - self._epoch_start
        self.epoch_history.append(epoch_seconds)

        val_acc = logs.get("val_accuracy")
        if val_acc is not None and (self.best_val_accuracy is None or val_acc > self.best_val_accuracy):
            self.best_val_accuracy = val_acc
            self.best_epoch = self._epoch

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "completed_epochs": self._epoch,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
        }, indent=2))
        os.replace(tmp, STATE_FILE)

        self._write(phase="training", force_log=True)
        avg_epoch = sum(self.epoch_history) / len(self.epoch_history)
        logging.info(
            f"epoch {self._epoch}/{self.total_epochs} done in {format_duration(epoch_seconds)} "
            f"(avg {format_duration(avg_epoch)}/epoch) -- loss {logs.get('loss', float('nan')):.4f} "
            f"acc {logs.get('accuracy', float('nan')):.4f} | val_loss {logs.get('val_loss', float('nan')):.4f} "
            f"val_acc {logs.get('val_accuracy', float('nan')):.4f} | best val_acc "
            f"{self.best_val_accuracy:.4f} @ epoch {self.best_epoch}"
        )

    def on_train_end(self, logs=None):
        self._write(phase="done", force_log=True)

    def _write(self, phase, log=True, force_log=False):
        self._last_write = time.time()
        now = datetime.now()
        epoch_elapsed = time.time() - self._epoch_start if self._epoch_start else 0.0
        step_rate = self._step / epoch_elapsed if epoch_elapsed > 0 else 0.0

        avg_epoch = sum(self.epoch_history) / len(self.epoch_history) if self.epoch_history else None
        remaining_epochs = self.total_epochs - self._epoch
        eta_epoch = (self.steps_per_epoch - self._step) / step_rate if step_rate > 0 else None
        eta_total = (
            (eta_epoch or 0) + max(remaining_epochs, 0) * avg_epoch if avg_epoch is not None else eta_epoch
        )

        status = {
            "phase": phase,
            "epoch": self._epoch,
            "epochs_total": self.total_epochs,
            "step": self._step,
            "steps_per_epoch": self.steps_per_epoch,
            "loss": self._logs.get("loss"),
            "accuracy": self._logs.get("accuracy"),
            "val_loss": self._logs.get("val_loss"),
            "val_accuracy": self._logs.get("val_accuracy"),
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
            "steps_per_sec": round(step_rate, 3),
            "avg_epoch_seconds": round(avg_epoch, 1) if avg_epoch is not None else None,
            "eta_epoch": format_duration(eta_epoch) if eta_epoch is not None else None,
            "eta_total": format_duration(eta_total) if eta_total is not None else None,
            "eta_total_finish": (
                (now + timedelta(seconds=eta_total)).isoformat(timespec="minutes")
                if eta_total is not None else None
            ),
            "started": self.started_at.isoformat(timespec="seconds"),
            "updated": now.isoformat(timespec="seconds"),
            "pid": os.getpid(),
        }
        tmp = PROGRESS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, PROGRESS_FILE)

        if log and (force_log or phase == "done"):
            logging.info(f"phase: {phase}")


def train_main(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, learning_rate: float = LEARNING_RATE,
                restart: bool = False):
    """Compile and fit WordClassifier on train/val, saving a checkpoint every epoch
    (checkpoints/word_classifier_latest.keras) and the best one by val_accuracy
    (checkpoints/word_classifier_best.keras). Resumes from the latest checkpoint unless
    restart=True or no checkpoint exists yet.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(LOG_FILE) for h in root_logger.handlers):
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(root_logger.handlers[0].formatter)
        root_logger.addHandler(file_handler)

    if restart:
        for f in (STATE_FILE, LATEST_CKPT, PROGRESS_FILE):
            f.unlink(missing_ok=True)

    train_ds, n_train, classes = build_dataset("train", batch_size, shuffle=True)
    val_ds, _, _ = build_dataset("val", batch_size)
    steps_per_epoch = -(-n_train // batch_size)  # ceil

    initial_epoch, best_val_accuracy, best_epoch = 0, None, None
    if STATE_FILE.exists() and LATEST_CKPT.exists():
        state = json.loads(STATE_FILE.read_text())
        initial_epoch = state["completed_epochs"]
        best_val_accuracy = state.get("best_val_accuracy")
        best_epoch = state.get("best_epoch")
        if initial_epoch >= epochs:
            logging.info(f"already completed {initial_epoch}/{epochs} epochs, nothing to do "
                         f"(pass --epochs with a higher number, or --restart, to continue)")
            return None
        logging.info(f"resuming from epoch {initial_epoch + 1}/{epochs} "
                     f"(best val_accuracy so far: {best_val_accuracy:.4f} @ epoch {best_epoch})")
        model = keras.models.load_model(LATEST_CKPT)
    else:
        model = WordClassifier(num_classes=len(classes))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(BEST_CKPT), monitor="val_accuracy", save_best_only=True,
                                         initial_value_threshold=best_val_accuracy),
        keras.callbacks.ModelCheckpoint(str(LATEST_CKPT), save_best_only=False),
        TrainingProgress(epochs, steps_per_epoch, best_val_accuracy, best_epoch),
    ]

    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, initial_epoch=initial_epoch,
                      callbacks=callbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--restart", action="store_true", help="ignore any existing checkpoint, start fresh")
    parser.add_argument("--status", action="store_true", help="print a snapshot of the progress dashboard")
    args = parser.parse_args()

    if args.status:
        from signai.word_classification.models.training_dashboard import show
        show(once=True)
    else:
        train_main(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                   restart=args.restart)
