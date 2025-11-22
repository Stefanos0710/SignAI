import json
import os
import shutil
from collections import Counter
import string

def get_gloss(class_list_file, gloss_id):
    """Liest die class_list-Datei und gibt das Gloss für die gegebene gloss_id zurück.
    Verwendet die übergebene Dateipfad-Variable (statt einer globalen Variable).
    """
    class_id = int(gloss_id)

    if not os.path.exists(class_list_file):
        raise FileNotFoundError(f"wlasl_class_list not found: {class_list_file}")

    with open(class_list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            # get id part
            try:
                current_id = int(parts[0])
            except ValueError:
                continue

            if current_id == class_id:
                gloss = " ".join(parts[1:]).strip()
                return gloss

    # didn´t find it
    return None


def create_dir_video(id, video_path, gloss, base_dir):
    # make sure that there is a gloss
    if not gloss:
        # if there is not gloss, skip this entry
        print(f"gloss is empty, skipping id {id}")
        return

    # copy video nur, wenn die Datei existiert
    if not os.path.exists(video_path):
        print(f"    didn´t found video: {video_path} (skipping entry {id})")
        return

    # data/raw_data dir
    raw_data_dir = os.path.join(base_dir, "data", "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    folder_name = f"wlasl_{id}"
    target_dir = os.path.join(raw_data_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    # build new videoname: gloss_id.mp4, z.B. BOOK_05237.mp4
    safe_gloss = str(gloss).replace("/", "-").replace("\\", "-")
    target_video_name = f"{safe_gloss}_{id}.mp4"
    target_video_path = os.path.join(target_dir, target_video_name)

    # copy video (hier existiert die Datei garantiert)
    shutil.copy2(video_path, target_video_path)

    # GLOSS.txt write file
    gloss_file_path = os.path.join(target_dir, "GLOSS.txt")
    with open(gloss_file_path, "w", encoding="utf-8") as gf:
        gf.write(str(gloss))


# helper: normalisieren eines worts (kleinschreibung, entferne punctuation an anfang/ende)
def _normalize_word(w):
    if not w:
        return ""
    # remove punctuation from both ends
    return w.strip().strip(string.punctuation).lower()


# ermittelt die Top-N am häufigsten vorkommenden Wörter aus der wlasl_class_list.txt
def get_top_words(class_list_path, top_n=50):
    """Gibt eine Liste von (wort, count)-Tupeln der top_n häufigsten Wörter zurück."""
    if not os.path.exists(class_list_path):
        raise FileNotFoundError(f"wlasl_class_list not found: {class_list_path}")

    counter = Counter()
    with open(class_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            gloss = " ".join(parts[1:])
            # split gloss into words and normalize
            for tok in gloss.split():
                nw = _normalize_word(tok)
                if nw:
                    counter[nw] += 1

    return counter.most_common(top_n)


# introduction message
print("=" * 80)
print(" WLASL Dataset to Raw Data Converter")
print("=" * 80)

# define paths early so top-mode can use them
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
class_list_path = os.path.join(BASE_DIR, "data", "wlasl", "wlasl_class_list.txt")

# ask user for number of classes to extract
print("\nPLease enter the number of classes you want to extract (100, 300, 1000, 2000 or top):")
num_classes = input("  => ")

# handle input: either one of the preset nslt files or 'top'
if num_classes == "top":
    print("\n")
    print("="*80)
    print(" Top words from WLASL Dataset to Raw Data Converter")
    print("="*80)
    print("\n Enter how many top words you want to use (e.g. 10, 50):")
    top_input = input("  => ")
    try:
        top_n = int(top_input)
        if top_n <= 0:
            raise ValueError()
    except ValueError:
        print("  ✗ Invalid number. Please enter a positive integer.")
        exit(1)
    # bestimme top-n wörter (jetzt mit counts)
    top_words_with_counts = get_top_words(class_list_path, top_n=top_n)
    top_words = set(w for w, _ in top_words_with_counts)
    print(f"Using top {len(top_words)} words (word:count): {top_words_with_counts[:20]}{('...' if len(top_words_with_counts)>20 else '')}")
    # choose the largest available nslt file automatically
    wlasl_dir = os.path.join(BASE_DIR, "data", "wlasl")
    nslt_files = [f for f in os.listdir(wlasl_dir) if f.startswith("nslt_") and f.endswith(".json")]
    if nslt_files:
        # pick the one with the largest number
        def _num_from_name(n):
            try:
                return int(n.split("nslt_")[1].split(".json")[0])
            except Exception:
                return -1
        nslt_chosen = max(nslt_files, key=_num_from_name)
        nslt = os.path.join(wlasl_dir, nslt_chosen)
    else:
        # fallback to full file
        nslt = os.path.join(wlasl_dir, "WLASL_v0.3.json")

elif num_classes in ["100", "300", "1000", "2000"]:
    nslt = os.path.join(BASE_DIR, "data", "wlasl", f"nslt_{num_classes}.json")
    top_words = None
else:
    print("  ✗ Invalid number of classes. Please enter one of: 100, 300, 1000, 2000 or 'top'")
    exit(1)

# file count
count = 0
processed = 0
skipped = 0

# make sure it the file exists
if not os.path.exists(nslt):
    print(f"  ✗ File not found: {nslt}")
    exit(1)

# load nslt_{num_classes}.json
with open(nslt, "r", encoding="utf-8") as f:
    print(f"\nReading file: {nslt}")
    nslt_data = json.load(f)

# go throuh every item
for action_id, info in nslt_data.items():
    subset = info.get("subset", "unknown")
    actions = info.get("action", [])

    # video path
    video_path = os.path.join(BASE_DIR, "data", "wlasl", "videos", f"{action_id}.mp4")

    # gloss
    gloss = None
    if actions:
        gloss_id = actions[0]
        gloss = get_gloss(class_list_path, gloss_id)    # if top_words is set, only process entries whose gloss contains at least one top word
    if top_words:
        if not gloss:
            print(f"Skipping {action_id}: no gloss available")
            skipped += 1
            count += 1
            continue
        # only accept exact match: the whole gloss (normalized) must equal one of the top words
        normalized_gloss = " ".join(_normalize_word(t) for t in str(gloss).split())
        if normalized_gloss not in top_words:
             # skip this entry
             skipped += 1
             count += 1
             continue

    print(f"Path: {video_path}, Subset: {subset}, Actions: {actions}, Gloss: {gloss}")

    # creating dir + file
    create_dir_video(action_id, video_path, gloss, BASE_DIR)

    count += 1
    processed += 1

print(f"\n✓ Processed {count} entries from {nslt} (created: {processed}, skipped: {skipped})")
"""
"05237": {"subset": "train", "action": [77, 1, 55]},
"69422": {"subset": "val", "action": [27, 1, 51]}
"""