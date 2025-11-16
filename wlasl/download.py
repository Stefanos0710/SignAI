import os
import shutil
import kagglehub

# Basis directory from this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Target directory for the dataset (project root /data folder)
TARGET_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir, "data", "wlasl"))

print("="*80)
print("WLASL Dataset Download from Kaggle")
print("="*80)
print()

# check for Kaggle credentials
kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
if not os.path.exists(kaggle_json):
    print("    WARNING: Kaggle credentials not found!")
    print(f"   Expected location: {kaggle_json}")
    print()
    print("   To set up Kaggle API credentials:")
    print("   1. Go to https://www.kaggle.com/settings/account")
    print("   2. Click 'Create New Token'")
    print("   3. Save kaggle.json to:", kaggle_json)
    print()
    print("   Continuing anyway (may fail if not authenticated)...")
    print()

try:
    print("Downloading WLASL dataset from Kaggle...")
    print("Dataset: risangbaskoro/wlasl-processed")
    print()
    print("This may take a while depending on your internet connection.")
    print("The dataset will be cached by Kaggle for future use.")
    print()

    # kagglehub downloads to its own cache directory
    download_path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")

    print()
    print("✓ Download completed!")
    print(f"  Kaggle cache location: {download_path}")
    print()

    # Create target directory
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Copy or create symlink to our data folder for easier access
    print(f"Setting up dataset in project directory: {TARGET_DIR}")

    # Check if target already exists
    if os.path.exists(TARGET_DIR) and os.listdir(TARGET_DIR):
        print(f"     Target directory already exists and is not empty")
        print(f"     Skipping copy. Dataset available at: {download_path}")
    else:
        print("  Copying files to project data directory...")
        # Copy files from download path to target directory
        for item in os.listdir(download_path):
            source = os.path.join(download_path, item)
            destination = os.path.join(TARGET_DIR, item)
            if os.path.isdir(source):
                if not os.path.exists(destination):
                    shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        print(f"  ✓ Files copied to: {TARGET_DIR}")

    print()
    print("="*80)
    print("DOWNLOAD SUCCESSFUL!")
    print("="*80)
    print()
    print(f"Dataset locations:")
    print(f"  • Kaggle cache: {download_path}")
    print(f"  • Project data:  {TARGET_DIR}")
    print()

    # List downloaded files
    print("Downloaded files:")
    for item in os.listdir(download_path):
        item_path = os.path.join(download_path, item)
        if os.path.isdir(item_path):
            file_count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
            print(f"  - {item}/ ({file_count} files)")
        else:
            size_mb = os.path.getsize(item_path) / (1024 * 1024)
            print(f"  - {item} ({size_mb:.2f} MB)")

except Exception as e:
    print()
    print("="*80)
    print("ERROR DURING DOWNLOAD")
    print("="*80)
    print(f"Error: {e}")
    print()
    print("Troubleshooting:")
    print("1. Make sure you have Kaggle API credentials set up")
    print("2. Make sure you have accepted the dataset terms on Kaggle")
    print("3. Visit: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed")
    print("4. Click 'Download' to accept terms (no need to actually download)")
    print()
    raise
