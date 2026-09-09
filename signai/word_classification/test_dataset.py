"""
in this file the dataset files
- test_data.npz / test_images.npz
- val_data.npz / val_images.npz
- train_data.npz / train_images.npz
(under dataset/processed/) are unpacked to inspect their content

for explizitly unpacking the files, the following code can be used:

 python test_dataset.py --all

or

 python test_dataset.py --test / --train / --val

To actually look at some of the hand/mouth crop images extracted alongside the keypoints,
dump a few of them to disk as .jpg (one crop type is one frame per sample by default):

 python test_dataset.py --images --train --count 5

writes dataset/processed/preview_images/train_<clip_id>_<left_hand|right_hand|mouth>.jpg,
which you can open with any image viewer.
"""

import argparse
import os

import cv2
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset", "processed")
PREVIEW_DIR = os.path.join(DATASET_DIR, "preview_images")
SPLITS = ["train", "val", "test"]
CROP_TYPES = ("left_hand", "right_hand", "mouth")


def inspect_split(split: str) -> None:
    path = os.path.join(DATASET_DIR, f"{split}_data.npz")
    if not os.path.exists(path):
        print(f"[{split}] missing: {path}")
        return

    data = np.load(path, allow_pickle=True)
    X, y, classes = data["X"], data["y"], data["classes"]

    print(f"[{split}] {path}")
    print(f"  X: shape={X.shape} dtype={X.dtype}")
    print(f"  y: shape={y.shape} dtype={y.dtype}")
    print(f"  classes: {len(classes)} labels")
    print(f"  label range: {int(y.min())}..{int(y.max())}")


def decode_crop(jpeg_bytes):
    """JPEG bytes -> BGR uint8 array, ready for cv2.imwrite.

    Duplicated (not imported) from preprocessing.decode_image_sequence: importing that
    module drags in mediapipe/tensorflow just to decode a jpeg, which is a bad trade for
    a quick inspection script.
    """
    return cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def last_content_frame(sequence) -> int:
    """Index of the last non-black frame in a decoded crop sequence.

    Clips are usually much shorter than MAX_FRAMES, so a fixed index (e.g. the middle
    of the 32-frame array) is often just padding. The real crops -- once a hand/mouth is
    first detected -- carry forward through the end of the clip's real frames, right
    before the black padding tail, so the last non-black frame is a representative one.
    Falls back to frame 0 if the hand/mouth was never detected at all (an all-black clip).
    """
    for idx in range(len(sequence) - 1, -1, -1):
        if decode_crop(sequence[idx]).any():
            return idx
    return 0


def dump_images(split: str, count: int, frame: int = None, out_dir: str = PREVIEW_DIR) -> None:
    """Write `count` samples' crops (one frame each) to `out_dir` as .jpg for eyeballing."""
    path = os.path.join(DATASET_DIR, f"{split}_images.npz")
    if not os.path.exists(path):
        print(f"[{split}] missing: {path}")
        return

    data = np.load(path, allow_pickle=True)
    clip_ids = data["clip_ids"]
    os.makedirs(out_dir, exist_ok=True)

    n = min(count, len(clip_ids))
    for i in range(n):
        for crop_type in CROP_TYPES:
            sequence = data[crop_type][i]
            idx = frame if frame is not None else last_content_frame(sequence)
            img = decode_crop(sequence[idx])
            out_path = os.path.join(out_dir, f"{split}_{clip_ids[i]}_{crop_type}.jpg")
            cv2.imwrite(out_path, img)
    print(f"[{split}] wrote {n} sample(s) x {len(CROP_TYPES)} crop type(s) -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="unpack train, val and test")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--images", action="store_true", help="dump crop previews instead of printing shapes")
    parser.add_argument("--count", type=int, default=5, help="how many samples to dump with --images (default 5)")
    parser.add_argument("--frame", type=int, default=None,
                         help="which frame index to dump (default: last non-black frame, since most "
                              "clips are much shorter than the padded 32-frame sequence)")
    args = parser.parse_args()

    selected = [s for s in SPLITS if getattr(args, s)]
    if args.all or not selected:
        selected = SPLITS

    for split in selected:
        if args.images:
            dump_images(split, args.count, args.frame)
        else:
            inspect_split(split)


if __name__ == "__main__":
    main()
