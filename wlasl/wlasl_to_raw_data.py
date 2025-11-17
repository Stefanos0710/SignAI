import json
import os
import shutil

def get_gloss(path, id):
    class_id = int(id)

    if not os.path.exists(class_list_path):
        raise FileNotFoundError(f"wlasl_class_list not found: {class_list_path}")

    with open(class_list_path, "r", encoding="utf-8") as f:
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


# introduction message
print("=" * 80)
print(" WLASL Dataset to Raw Data Converter")
print("=" * 80)

# ask user for number of classes to extract
print("\nPLease enter the number of classes you want to extract (100, 300, 1000, 2000):")
num_classes = input("  => ")

# validate input
valid = num_classes in ["100", "300", "1000", "2000"]
if not valid:
    print("  ✗ Invalid number of classes. Please enter one of: 100, 300, 1000, 2000")
    exit(1)

# define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
nslt = os.path.join(BASE_DIR, 'data', 'wlasl', f'nslt_{num_classes}.json')
class_list_path = os.path.join(BASE_DIR, "data", "wlasl", "wlasl_class_list.txt")

# file count
count = 0

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
    video_path = os.path.join(BASE_DIR, 'data', 'wlasl', 'videos', f'{action_id}.mp4')

    # gloss
    gloss = None
    if actions:
        gloss_id = actions[0]
        gloss = get_gloss(class_list_path, gloss_id)

    print(f"Path: {video_path}, Subset: {subset}, Actions: {actions}, Gloss: {gloss}")
    count += 1

print(f"\n✓ Processed {count} entries from {nslt}")
"""
"05237": {"subset": "train", "action": [77, 1, 55]},
"69422": {"subset": "val", "action": [27, 1, 51]}
"""