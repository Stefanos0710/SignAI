import json
import os
import shutil

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

# make sure it the file exists
if not os.path.exists(nslt):
    print(f"  ✗ File not found: {nslt}")
    exit(1)

# load nslt_{num_classes}.json
with open(nslt, "r", encoding="utf-8") as f:
    print(f"Reading file: {nslt}")
    nslt_data = json.load(f)

# go throuh every item
for action_id, info in nslt_data.items():
    subset = info.get("subset", "unknown")
    actions = info.get("action", [])
    print(f'"{action_id}": {{"subset": "{subset}", "action": {actions}}},')

"""
"05237": {"subset": "train", "action": [77, 1, 55]},
"69422": {"subset": "val", "action": [27, 1, 51]}
"""