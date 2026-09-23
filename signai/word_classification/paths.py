"""Configurable storage paths for the word-classification dataset."""

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
