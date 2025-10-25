from pathlib import Path
import zipfile
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_DIR = PROJECT_ROOT / "app" / "builds" / "zip"

# make zip folder if not there
os.makedirs(ZIP_DIR, exist_ok=True)

def zip_items(items, zip_name):
    # start make zip file
    print("Starting to create zip file...")

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in items:
            item_path = PROJECT_ROOT / Path(item)

            if not item_path.exists():
                print(f"Warning: {item_path.resolve()} not here or skip.")
                continue

            if item_path.is_dir():
                print(f"Do directory: {item_path}")
                for file in item_path.rglob('*'):
                    if file.is_file():
                        relative_path = file.relative_to(PROJECT_ROOT)
                        zipf.write(file, relative_path)
                        print(f"\t{relative_path}")
                    else:
                        print(f"\tSkip directory: {file.relative_to(PROJECT_ROOT)}")

            elif item_path.is_file():
                relative_path = item_path.relative_to(PROJECT_ROOT)
                zipf.write(item_path, relative_path)
                print(f"Do file: {relative_path}")
            else:
                print(f"Skip unknown thing: {item_path.resolve()}")

    print(f"Zip file make: {zip_name}")

def get_current_version():
    version_file = PROJECT_ROOT / "app" / "updater" / "version.txt"
    if not version_file.exists():
        print(f"Version file not here: {version_file.resolve()}")
        return "No_version"
    with open(version_file, "r") as f:
        return f.read().strip()

items = [
    "app/data",
    "app/settings",
    "app/updater",
    "app/dist/SignAI - Desktop.exe"
]

if __name__ == "__main__":
    zip_name = ZIP_DIR / f"{get_current_version()}.zip"
    zip_items(items, zip_name)
