"""Configurable storage paths for the word-classification dataset and model checkpoints."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path):
    """Load simple KEY=VALUE settings without overriding the shell environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env_file(REPO_ROOT / ".env")

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "dataset"
DATASET_DIR = Path(os.getenv("SIGNAI_WORD_DATASET_DIR", DEFAULT_DATASET_DIR)).expanduser()
PROCESSED_DIR = Path(
    os.getenv("SIGNAI_WORD_PROCESSED_DIR", DATASET_DIR / "processed")
).expanduser()

# Every trained model gets its own checkpoints/v<N>/ (checkpoints, state.json, progress.json,
# train.log, eval/) -- bump this for each new training run. v40: per-clip split; v41: first
# person split (split_by_person.py). SIGNAI_WORD_MODEL_VERSION=40 points the scripts at an older one.
MODEL_VERSION = int(os.getenv("SIGNAI_WORD_MODEL_VERSION", "41"))
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / f"v{MODEL_VERSION}"
