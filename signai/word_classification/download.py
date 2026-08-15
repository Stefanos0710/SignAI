"""
Downloads the Public DGS Corpus (MEINE DGS - annotiert) into
signai/word_classification/dataset.

Run from the repo root:
    python signai/word_classification/download.py                  # eaf + openpose
    python signai/word_classification/download.py --parts eaf      # annotations only
    python signai/word_classification/download.py --limit 5        # small trial
    python signai/word_classification/download.py --parts videos   # ~hundreds of GB

The file list is scraped from the official transcript index page; every
transcript ID gets its ELAN annotation (.eaf) and OpenPose keypoints
(.json.gz) by default -- that is what word-level training needs.

Data is provided by the DGS-Korpus project for linguistic research only.
See https://www.sign-lang.uni-hamburg.de/meinedgs/ling/license_en.html
"""

import argparse
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

BASE_URL = "https://www.sign-lang.uni-hamburg.de/meinedgs"
INDEX_URL = f"{BASE_URL}/ling/start-name_en.html"
DATASET_FOLDER = "signai/word_classification/dataset"

# parts of the corpus that can be downloaded, with a rough per-file size so the
# script can warn about the total before spending hours on it
PARTS = {
    "eaf": 0.7,        # ELAN annotations (glosses + time codes)
    "openpose": 92.0,  # OpenPose keypoints, gzipped json
    "ilex": 4.0,       # iLex annotations
    "srt": 0.02,       # subtitles
    "cmdi": 0.02,      # metadata
    "videos": 60.0,    # mp4, several camera angles per transcript
}
DEFAULT_PARTS = ["eaf", "openpose"]

LINK_RE = re.compile(r'href="\.\./(%s)/([^"]+)"' % "|".join(PARTS))


def fetch_index(dataset_folder):
    """Returns the index HTML, cached on disk so re-runs don't refetch it."""
    cache = os.path.join(dataset_folder, "start-name_en.html")

    if os.path.exists(cache):
        logging.info("Using cached transcript index")
        with open(cache, encoding="utf-8") as f:
            return f.read()

    logging.info(f"Fetching transcript index: {INDEX_URL}")
    response = requests.get(INDEX_URL, timeout=60)
    response.raise_for_status()
    html = response.content.decode("utf-8", "replace")

    with open(cache, "w", encoding="utf-8") as f:
        f.write(html)

    return html


def collect_files(html, parts, limit=None):
    """Extracts (part, relative_path) pairs from the index, de-duplicated."""
    files, seen = [], set()

    for part, path in LINK_RE.findall(html):
        if part not in parts or (part, path) in seen:
            continue
        seen.add((part, path))
        files.append((part, path))

    if limit:
        # keep the first `limit` transcripts per part, not the first N files
        kept, counts = [], {}
        for part, path in files:
            ids = counts.setdefault(part, [])
            transcript = path.split("/")[0].split("_")[0]
            if transcript not in ids:
                if len(ids) >= limit:
                    continue
                ids.append(transcript)
            kept.append((part, path))
        files = kept

    return files


def download_file(part, path, dataset_folder):
    """Downloads one file, resuming a partial `.part` file if present."""
    url = f"{BASE_URL}/{part}/{path}"
    out = os.path.join(dataset_folder, part, path.replace("/", os.sep))
    tmp = out + ".part"

    if os.path.exists(out):
        return 0

    os.makedirs(os.path.dirname(out), exist_ok=True)

    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={done}-"} if done else {}

    response = requests.get(url, headers=headers, stream=True, timeout=60)

    if done and response.status_code != 206:
        # server ignored the range request, start over
        done = 0
    response.raise_for_status()

    with open(tmp, "ab" if done else "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            done += len(chunk)

    os.replace(tmp, out)
    return done


def download_all(files, dataset_folder, workers):
    downloaded = failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_file, part, path, dataset_folder): (part, path)
            for part, path in files
        }

        with tqdm(total=len(futures), desc="Downloading", unit="file") as bar:
            for future in as_completed(futures):
                part, path = futures[future]
                try:
                    downloaded += future.result()
                except Exception as e:
                    failed += 1
                    logging.error(f"Failed {part}/{path}: {e}")
                bar.update(1)

    return downloaded, failed


def folder_size_mb(folder):
    total = 0
    for root, _, names in os.walk(folder):
        for name in names:
            total += os.path.getsize(os.path.join(root, name))
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Download the Public DGS Corpus")
    parser.add_argument(
        "--parts",
        default=",".join(DEFAULT_PARTS),
        help=f"comma-separated subset of {', '.join(PARTS)} (default: %(default)s)",
    )
    parser.add_argument("--limit", type=int, help="only the first N transcripts per part")
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads (default: 4)")
    parser.add_argument("--out", default=DATASET_FOLDER, help="target folder (default: %(default)s)")
    args = parser.parse_args()

    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        parser.error(f"unknown part(s): {', '.join(unknown)}. Choose from {', '.join(PARTS)}")

    os.makedirs(args.out, exist_ok=True)

    files = collect_files(fetch_index(args.out), parts, args.limit)
    estimate_gb = sum(PARTS[part] for part, _ in files) / 1024

    print("\n" + "=" * 70)
    print("  Public DGS Corpus Downloader  ")
    print("=" * 70)
    print("Dataset Name      : MEINE DGS - annotiert (Public DGS Corpus)")
    print("Description       : German Sign Language corpus with gloss annotations")
    print("                    and OpenPose keypoints, for linguistic research")
    print(f"Index URL         : {INDEX_URL}")
    print(f"Target Folder     : {args.out}")
    print(f"Parts             : {', '.join(parts)}")
    print(f"Files             : {len(files)}")
    print(f"Estimated size    : ~{estimate_gb:.1f} GB")
    print("Notes             :")
    print("  - Already downloaded files are skipped, interrupted ones resume")
    print("  - Research use only, see the license page linked in this script")
    print("=" * 70 + "\n")

    start = time.time()
    downloaded, failed = download_all(files, args.out, args.workers)
    duration = time.time() - start

    print("\n" + "=" * 70)
    print("  Download Completed  ")
    print("=" * 70)
    print(f"Dataset Root Folder : {args.out}")
    print(f"Downloaded this run : {downloaded / (1024 ** 3):.2f} GB")
    print(f"Total Size          : {folder_size_mb(args.out):.2f} MB")
    print(f"Duration            : {duration:.1f} seconds")
    print(f"Failed files        : {failed}")
    print("=" * 70)

    if failed:
        logging.warning("Re-run the script to retry the failed files.")


def demo():
    """Self-check on the index parsing, run with: python download.py --self-check"""
    html = (
        '<a href="../eaf/1413485.eaf" download="1413485.eaf">'
        '<a href="../openpose/1413485_openpose.json.gz" download="x">'
        '<a href="../videos/1413485/1413485_1a1.mp4" download="x">'
        '<a href="../eaf/1413485.eaf" download="1413485.eaf">'
        '<a href="../graphics/video.png">'
        '<a href="../eaf/1413451-11105600-11163240.eaf" download="x">'
    )

    assert collect_files(html, ["eaf"]) == [
        ("eaf", "1413485.eaf"),
        ("eaf", "1413451-11105600-11163240.eaf"),
    ], "must de-duplicate and ignore other parts/graphics"

    assert collect_files(html, ["eaf", "openpose", "videos"]) == [
        ("eaf", "1413485.eaf"),
        ("openpose", "1413485_openpose.json.gz"),
        ("videos", "1413485/1413485_1a1.mp4"),
        ("eaf", "1413451-11105600-11163240.eaf"),
    ], "must keep every requested part, including nested video paths"

    assert collect_files(html, ["eaf"], limit=1) == [("eaf", "1413485.eaf")], (
        "limit must count transcripts, not files"
    )

    print("self-check passed")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        demo()
    else:
        main()
