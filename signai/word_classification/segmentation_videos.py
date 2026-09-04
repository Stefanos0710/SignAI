"""
In this script, the downloaded dataset will be segmented into individual videos, each containing a single word.
The segmentation is based on the time codes provided in the ELAN annotations (EAF files).

Input:
 - dataset/
    - eaf/
        - ...
    videos/
        - ...

Output:

csv file with all segments

path, word, start_time, end_time, transcript, gloss

individual per-word clips, cut out of the sentence videos with ffmpeg (must
be on PATH), written to dataset/word_clips/<eaf_stem>_<participant>_<index>.mp4

"""
import csv
import logging
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pympi
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BATCH_SIZE = 150   # segments per ffmpeg call; keeps the command line under Windows' ~32K limit
WORKERS = 4        # parallel ffmpeg processes (subprocess calls release the GIL)


def clean_trancript(transcript):
    """Clean the transcript by removing unwanted characters and formatting."""
    transcript = transcript.lstrip('$')                # leading gesture/pointer marker
    transcript = re.sub(r'[\^\*]+$', '', transcript)    # removing ^ / * markers
    transcript = re.sub(r'\d+[A-Z]?$', '', transcript)  # removing variant number/letter

    return transcript

def create_csv_from_eaf(eaf_path, participants):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eaf_file", "participant", "index", "start_ms", "end_ms", "transcript"])

        for eaf_path in sorted(eaf_dir.glob("*.eaf")):
            eaf = pympi.Elan.Eaf(str(eaf_path))
            tier_names = eaf.get_tier_names()

            found_any = False
            for participant in participants:
                tier_name = f"Lexem_Gebärde_r_{participant}"
                if tier_name not in tier_names:
                    continue
                found_any = True

                annotations = eaf.get_annotation_data_for_tier(tier_name)
                for i, (start, end, transcription) in enumerate(annotations, start=1):
                    writer.writerow([eaf_path.name, participant, i, start, end, clean_trancript(transcription)])

            if not found_any:
                print(f"skipping {eaf_path.name}: no 'Lexem_Gebärde_r_A/B' tier")

        print(f"wrote segments to {output_csv}")


def pick_video_path(videos_dir, transcript_id, participant):
    """Pick the camera-specific video file for a participant.

    Participant A is `<id>_1a1.mp4`, falling back to the bare `<id>.mp4` for
    older transcripts that were only ever recorded with a single camera.
    Participant B is always `<id>_1b1.mp4`. The `_1c` file / bare wide file
    on newer transcripts are two-signer overview shots and are never used.
    """
    folder = videos_dir / transcript_id
    suffix = "1a1" if participant == "A" else "1b1"
    candidate = folder / f"{transcript_id}_{suffix}.mp4"
    if candidate.exists():
        return candidate
    if participant == "A":
        bare = folder / f"{transcript_id}.mp4"
        if bare.exists():
            return bare
    return None


def load_segments(segments_csv):
    """Group segment rows by (eaf_file, participant) -> [(index, start_ms, end_ms), ...]."""
    groups = defaultdict(list)
    with open(segments_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            start, end = int(row["start_ms"]), int(row["end_ms"])
            if end <= start:
                continue
            groups[(row["eaf_file"], row["participant"])].append((int(row["index"]), start, end))
    return groups


def cut_batch(video_path, batch, out_dir, prefix):
    """Cut one batch of segments out of a single source video in one ffmpeg process.

    Chaining multiple `-ss/-t/output` triplets after a single `-i` lets ffmpeg
    produce many trimmed clips from one file without reopening/redecoding it
    per clip -- far cheaper than one ffmpeg process (or a Python decode loop)
    per segment.
    """
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path)]
    for index, start_ms, end_ms in batch:
        out = out_dir / f"{prefix}_{index:04d}.mp4"
        cmd += [
            "-ss", f"{start_ms / 1000:.3f}",
            "-t", f"{(end_ms - start_ms) / 1000:.3f}",
            "-an",  # no audio needed for keypoint extraction
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            str(out),
        ]
    subprocess.run(cmd, check=True, capture_output=True)


def cutting_segments_from_videos(segments_csv, videos_dir, out_dir, workers=WORKERS):
    """Cut every segment in `segments_csv` into its own clip under `out_dir`.

    Resumable: segments whose output file already exists are skipped, as are
    (eaf_file, participant) groups whose video hasn't been downloaded yet.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH -- install it (e.g. `conda install ffmpeg`) first.")

    out_dir.mkdir(parents=True, exist_ok=True)
    groups = load_segments(segments_csv)

    jobs = []
    missing_videos = 0
    for (eaf_file, participant), segs in groups.items():
        transcript_id = Path(eaf_file).stem
        video_path = pick_video_path(videos_dir, transcript_id, participant)
        if video_path is None:
            missing_videos += 1
            continue

        prefix = f"{transcript_id}_{participant}"
        segs.sort(key=lambda s: s[1])  # chronological, so each batch decodes forward-only
        pending = [s for s in segs if not (out_dir / f"{prefix}_{s[0]:04d}.mp4").exists()]

        for start in range(0, len(pending), BATCH_SIZE):
            jobs.append((video_path, pending[start:start + BATCH_SIZE], prefix))

    if missing_videos:
        logging.info(f"skipping {missing_videos} eaf/participant group(s): video not downloaded")

    if not jobs:
        logging.info("nothing to cut (all clips already exist or no videos downloaded)")
        return

    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(cut_batch, video_path, batch, out_dir, prefix): prefix
            for video_path, batch, prefix in jobs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Cutting clips"):
            prefix = futures[fut]
            try:
                fut.result()
            except subprocess.CalledProcessError as e:
                failed += 1
                err = e.stderr.decode("utf-8", "replace").strip() if e.stderr else str(e)
                logging.error(f"failed batch for {prefix}: {err}")

    logging.info(f"cut clips for {len(jobs) - failed}/{len(jobs)} batch(es) into {out_dir}")


def demo():
    """Self-check for pick_video_path against real downloaded folders, run with:
    python segmentation_videos.py --self-check
    """
    cases = [
        ("1413446", "A", "1413446.mp4"),  # no _1a1 file on this transcript, falls back to bare
        ("1413446", "B", "1413446_1b1.mp4"),
        ("1418836-15524810-15550340", "A", "1418836-15524810-15550340_1a1.mp4"),
        ("1418836-15524810-15550340", "B", "1418836-15524810-15550340_1b1.mp4"),
    ]

    ran = 0
    for transcript_id, participant, expected_name in cases:
        if not (videos_dir / transcript_id).exists():
            continue  # sample not downloaded on this machine, skip
        picked = pick_video_path(videos_dir, transcript_id, participant)
        assert picked is not None, f"expected a video for {transcript_id}/{participant}"
        assert picked.name == expected_name, (
            f"{transcript_id}/{participant}: got {picked.name}, expected {expected_name}"
        )
        ran += 1

    assert pick_video_path(videos_dir, "does-not-exist", "A") is None, "missing folder must return None"

    print(f"self-check passed ({ran} real folder case(s) checked)")


def main():
    create_csv_from_eaf(eaf_dir, participants)
    cutting_segments_from_videos(output_csv, videos_dir, clips_dir)


# set up paths of dataset and output files
dataset_dir = Path(__file__).resolve().parent / "dataset"
eaf_dir = dataset_dir / "eaf"
videos_dir = dataset_dir / "videos"
clips_dir = dataset_dir / "word_clips"
output_csv = dataset_dir / "segments.csv"

participants = ["A", "B"]

if __name__ == "__main__":
    if "--self-check" in sys.argv:
        demo()
    else:
        main()
