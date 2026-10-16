"""Re-split the already preprocessed word dataset by person (signer-independent), without
re-extracting keypoints/crops or re-running DINOv3: every clip's rows are pooled by clip_id from
the existing processed/{train,val,test,excluded}_*.npz files and written back out under
preprocessing.assign_person_splits -- the same split preprocessing.py itself now writes.

Why: the old split was per clip (stratified per gloss), so every signer had clips in train, val
and test, and val/test only measured within-signer accuracy. Now every signer, together with
their recording partner, is in exactly one split, so val/test are unseen-signer numbers.

Files re-split (all families, so they stay row-aligned by clip_ids as the loaders assert):
    {split}_data.npz            X, y, clip_ids, classes, signers (new: the informant code per clip)
    {split}_images.npz          clip_ids + left_hand/right_hand/mouth JPEG crops
    {split}_hand_features.npz   cached DINOv3 hand features (hand_stream.py), if present
    {split}_face_features.npz   cached DINOv3 mouth features (face_stream.py), if present
    split.json                  which signers ended up where
    split_before_<time>.npz     the clip_ids each split had before, as a record of the old split
Nothing is lost: excluded_* holds the clips outside the new train vocabulary, so every clip keeps
all its data and running this again (e.g. with another --seed) works from the same pool.

Run from the repo root:
    python signai/word_classification/models/split_by_person.py --dry-run    # stats only, writes nothing
    python signai/word_classification/models/split_by_person.py

Each family's new files are written next to the old ones and swapped in only once complete and
checked, so an interruption leaves the old files in place. That needs as much free disk as the
largest family (the crops, ~27 GB); it's checked before anything is written. Memory: one array
pooled over all splits at a time, ~10 GB at most.

Afterwards train from scratch -- python signai/word_classification/models/train.py --restart --
a checkpoint from the old split has already seen the new val/test signers.
"""
import argparse
import logging
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from signai.word_classification.preprocessing import (  # noqa: E402
    CROP_TYPES, EXCLUDED, MIN_SAMPLES_PER_CLASS, SEED, SPLIT_NAMES, assign_person_splits, class_lists,
    list_jobs, load_signers, split_rows, transcript_of, write_split_info,
)
from signai.word_classification.paths import PROCESSED_DIR  # noqa: E402

SOURCES = (*SPLIT_NAMES, EXCLUDED)
# family -> (array members carried over row by row, zip compression of the original writer)
FAMILIES = {
    "data": (("X",), zipfile.ZIP_DEFLATED),  # y/classes/signers are rebuilt, see main()
    "hand_features": (("left_hand", "right_hand"), zipfile.ZIP_STORED),
    "face_features": (("mouth",), zipfile.ZIP_STORED),
    "images": (CROP_TYPES, zipfile.ZIP_STORED),
}
OPTIONAL = ("hand_features", "face_features")  # only exist once hand_stream/face_stream ran
MARGIN = 1 << 30  # bytes of free disk kept on top of the estimate


def family_path(split, family, folder=PROCESSED_DIR):
    return Path(folder) / f"{split}_{family}.npz"


def present_files(family):
    """-> {split: path} for the family's files that exist."""
    return {s: family_path(s, family) for s in SOURCES if family_path(s, family).exists()}


def load_member(path, key):
    with np.load(path, allow_pickle=True) as f:
        return f[key]


def write_member(zf, key, array):
    """One array as one .npy member -- what np.savez does per keyword, but one member at a time,
    so a family never has to be in memory all at once."""
    with zf.open(f"{key}.npy", "w", force_zip64=True) as f:
        np.lib.format.write_array(f, np.asanyarray(array), allow_pickle=True)


def positions_of(pool_ids, wanted):
    """Row of every id of `wanted` in `pool_ids`, checking that each is there exactly once."""
    sorter = np.argsort(pool_ids, kind="stable")
    found = sorter[np.minimum(np.searchsorted(pool_ids, wanted, sorter=sorter), len(pool_ids) - 1)]
    if len(np.unique(pool_ids)) != len(pool_ids) or not np.array_equal(pool_ids[found], wanted):
        raise RuntimeError("clip_ids are missing or duplicated across the split files")
    return found


def load_pool():
    """clip_ids, label strings and current split of every clip, pooled over all *_data.npz."""
    files = present_files("data")
    if not files:
        raise RuntimeError(f"no *_data.npz in {PROCESSED_DIR} -- run preprocessing.py first")
    clip_ids, labels, old_split = [], [], []
    for split, path in files.items():
        with np.load(path, allow_pickle=True) as f:
            clip_ids.append(f["clip_ids"])
            labels.append(f["classes"][f["y"]])
            old_split.append(np.full(len(f["y"]), split))
    clip_ids = np.concatenate(clip_ids)
    if len(np.unique(clip_ids)) != len(clip_ids):
        raise RuntimeError("a clip_id is in more than one *_data.npz -- an interrupted re-split? "
                           "restore the old files from the backup dir")
    return clip_ids, np.concatenate(labels), np.concatenate(old_split)


def clip_metadata(clip_ids, labels):
    """-> (signers, transcripts) per clip. clip_id is a clip's index in preprocessing.list_jobs()'s
    sorted list; checked against the stored labels so a changed word_clips/ or segments.csv fails
    loudly instead of silently giving clips the wrong signer."""
    jobs = list_jobs()
    if clip_ids.max() >= len(jobs):
        raise RuntimeError("clip_ids point past the clips in word_clips/ -- clips were removed since preprocessing")
    mismatched = sum(jobs[cid][1] != label for cid, label in zip(clip_ids.tolist(), labels.tolist()))
    if mismatched:
        raise RuntimeError(f"{mismatched:,} clip_id(s) no longer match their clip's label -- word_clips/ or "
                           "segments.csv changed since preprocessing.py ran")
    names = [jobs[cid][0].name for cid in clip_ids.tolist()]
    signer_of = load_signers()
    return np.array([signer_of[name] for name in names]), [transcript_of(name) for name in names]


def report(split, signer_split, signers, labels, old_split):
    """Log the split: clips/signers/regions/classes per split, what's excluded and why, and how
    much of the new val/test the old per-clip split had in train."""
    region = lambda signer: signer.split("-")[0]  # noqa: E731
    all_regions = {region(s) for s in signer_split}
    vocabulary = class_lists(labels, split)["train"]
    logging.info(f"{len(split):,} clips, {len(signer_split)} signers, {len(all_regions)} regions | vocabulary: "
                 f"{len(vocabulary):,} classes with >= {MIN_SAMPLES_PER_CLASS} train clips")
    for name in SPLIT_NAMES:
        mask = split == name
        who = [s for s, n in signer_split.items() if n == name]
        line = (f"{name:>5}: {mask.sum():>7,} clips ({mask.mean():5.1%}) | {len(who):>3} signers from "
                f"{len({region(s) for s in who}):>2}/{len(all_regions)} regions | "
                f"{len(set(labels[mask].tolist())):>5,} classes")
        if name != "train":
            line += f" | {np.mean(old_split[mask] == 'train'):.0%} of these clips were in the old train split"
        logging.info(line)
    excluded = split == EXCLUDED
    home = np.array([signer_split[s] for s in signers.tolist()])
    logging.info(f"{EXCLUDED}: {excluded.sum():,} clips ({excluded.mean():.1%}) -- "
                 f"{(excluded & (home == 'train')).sum():,} train clips of glosses with < {MIN_SAMPLES_PER_CLASS} "
                 f"train clips, {(excluded & (home != 'train')).sum():,} val/test clips of glosses outside the "
                 "vocabulary")


def families_to_resplit(clip_ids):
    """The families present, after checking each covers exactly the clips of *_data.npz -- before
    anything is touched, so an incomplete family can't leave a half re-split dataset behind."""
    data_splits = set(present_files("data"))
    families = []
    for family in FAMILIES:
        files = present_files(family)
        if not files and family in OPTIONAL:
            logging.info(f"{family}: none cached yet, skipping")
            continue
        if set(files) != data_splits:
            missing = sorted(data_splits - set(files))
            raise RuntimeError(f"{family}: missing for {missing} (have {sorted(files)}) -- "
                               + (f"compute them first (--extract-features <split>) or delete the {family} files"
                                  if family in OPTIONAL else "the processed dataset is incomplete"))
        pooled = np.concatenate([load_member(p, "clip_ids") for p in files.values()])
        if len(pooled) != len(clip_ids) or not np.array_equal(np.sort(pooled), np.sort(clip_ids)):
            raise RuntimeError(f"{family}: clip_ids differ from *_data.npz's -- stale {family} files?")
        families.append(family)
    return families


def check_space(families):
    """A family's new files sit next to its old ones until swapped in, so the largest family has
    to fit into the free space once more."""
    sizes = {f: sum(p.stat().st_size for p in present_files(f).values()) for f in families}
    family = max(sizes, key=sizes.get)
    free = shutil.disk_usage(PROCESSED_DIR).free
    if free < sizes[family] + MARGIN:
        raise RuntimeError(f"{PROCESSED_DIR}: the new {family} files need ~{(sizes[family] + MARGIN) / 2**30:.1f} "
                           f"GiB free, there are {free / 2**30:.1f} GiB -- free up "
                           f"{(sizes[family] + MARGIN - free) / 2**30:.1f} GiB on that disk first")


def resplit_family(family, target_ids, extra):
    """Write {split}_{family}.npz for every target split with exactly target_ids[split]'s rows.
    extra: {split: {member: array}} written as is (data's y/classes/signers)."""
    members, compression = FAMILIES[family]
    sources = present_files(family)
    pooled_ids = np.concatenate([load_member(p, "clip_ids") for p in sources.values()])
    positions = {name: positions_of(pooled_ids, ids) for name, ids in target_ids.items()}
    tmp = {name: family_path(name, family).with_suffix(".npz.tmp") for name in target_ids}
    zips = {name: zipfile.ZipFile(path, "w", compression, allowZip64=True) for name, path in tmp.items()}
    try:
        for name, ids in target_ids.items():
            write_member(zips[name], "clip_ids", ids)
            for key, value in extra.get(name, {}).items():
                write_member(zips[name], key, value)
        for key in members:
            logging.info(f"{family}: {key}")
            pooled = np.concatenate([load_member(p, key) for p in sources.values()])
            for name, pos in positions.items():
                write_member(zips[name], key, pooled[pos])
            del pooled
    except BaseException:
        for z in zips.values():
            z.close()
        for path in tmp.values():
            path.unlink(missing_ok=True)
        raise
    for z in zips.values():
        z.close()

    for name, path in tmp.items():
        if not np.array_equal(load_member(path, "clip_ids"), target_ids[name]):
            raise RuntimeError(f"{path}: written clip_ids don't match")
        os.replace(path, family_path(name, family))
    logging.info(f"{family}: done ({', '.join(f'{n} {len(i):,}' for n, i in target_ids.items())})")


def main(dry_run=False, seed=SEED):
    clip_ids, labels, old_split = load_pool()
    signers, transcripts = clip_metadata(clip_ids, labels)
    split, signer_split = assign_person_splits(signers, transcripts, labels, seed=seed)
    report(split, signer_split, signers, labels, old_split)
    if dry_run:
        logging.info("--dry-run: nothing written")
        return

    families = families_to_resplit(clip_ids)
    check_space(families)
    record = PROCESSED_DIR / f"split_before_{datetime.now():%Y%m%d-%H%M%S}.npz"
    np.savez(record, **{name: clip_ids[old_split == name] for name in SOURCES})
    logging.info(f"old split's clip_ids -> {record.name}")

    rows = split_rows(split, clip_ids, seed)
    classes = class_lists(labels, split)
    target_ids = {name: clip_ids[idx] for name, idx in rows.items()}
    data_extra = {
        name: {"y": np.searchsorted(classes[name], labels[idx]).astype(np.int64),
               "classes": classes[name], "signers": signers[idx]}
        for name, idx in rows.items()
    }
    for family in families:
        resplit_family(family, target_ids, data_extra if family == "data" else {})

    write_split_info(split, signer_split, signers, len(classes["train"]), PROCESSED_DIR, seed)
    logging.info(f"done -- signers per split in {PROCESSED_DIR / 'split.json'}. Train from scratch: "
                 "python signai/word_classification/models/train.py --restart")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="only print what the split would look like")
    parser.add_argument("--seed", type=int, default=SEED, help=f"shuffles which signers go where (default {SEED})")
    args = parser.parse_args()
    main(dry_run=args.dry_run, seed=args.seed)
