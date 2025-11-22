import gzip
import pickle
import os
import shutil


def read_gzip_pickle(file_path):
    print("Load gzip pickle file...")
    with gzip.open(file_path, "rb") as f:
        data = pickle.load(f)
    print("✓ Pickle-GZIP loaded successfully.")
    print(f"Loaded data: {len(data)}")
    return data

def join_gloss_field(gloss_field):
    if gloss_field is None:
        return ""
    # if it's already a string
    if isinstance(gloss_field, str):
        return gloss_field.strip()
    # if it's a list or iterable of tokens
    try:
        tokens = list(gloss_field)
    except Exception:
        return str(gloss_field).strip()

    # normalize tokens to strings and strip
    norm_tokens = [str(t).strip() for t in tokens if t is not None]
    if not norm_tokens:
        return ""

    # if most tokens are single characters join without spaces
    single_char_count = sum(1 for t in norm_tokens if len(t) == 1)
    if single_char_count >= len(norm_tokens) and len(norm_tokens) > 1:
        return ''.join(norm_tokens)

    # otherwise join with spaces
    return ' '.join(norm_tokens)


# path to gzip pickle and video base
gzip_path = r"C:\Users\stefa\Documents\GitHub\SignAI\data\PHOENIX-Weather-2014T-compressed\phoenix14t.pami0.train.annotations_only.gzip"
video_base = r"C:\Users\stefa\Documents\GitHub\SignAI\data\PHOENIX-Weather-2014T-compressed\videos_phoenix\videos"

# get base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
raw_data_dir = os.path.join(BASE_DIR, "data", "raw_data_phoenix")
os.makedirs(raw_data_dir, exist_ok=True)

# load gzip pickle
obj = read_gzip_pickle(gzip_path)

# process entries
processed = 0
missing_videos = 0

for i, entry in enumerate(obj):
    # name from file
    name = entry.get("name")
    gloss_list = entry.get("gloss", [])

    # join gloss field
    gloss_str = join_gloss_field(gloss_list)

    # path to the video
    video_path = os.path.join(video_base, name + ".mp4")

    # check if it exists
    if not os.path.exists(video_path):
        print(f"✗ Video missing: {name}.mp4")
        missing_videos += 1
        continue

    # dir for samples
    folder_name = f"train_data"
    target_dir = os.path.join(raw_data_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    # copy videos
    target_video = os.path.join(target_dir, "video.mp4")
    shutil.copy2(video_path, target_video)

    # save gloss
    gloss_file = os.path.join(target_dir, "GLOSS.txt")
    with open(gloss_file, "w", encoding="utf-8") as f:
        f.write(gloss_str)

    print(f"✓ {folder_name}: {gloss_str}")
    processed += 1

print("\n--------------------------------------")
print(f"Finished!")
print(f"Proceed videos: {processed}")
print(f"Missing Videos: {missing_videos}")
print("--------------------------------------")
