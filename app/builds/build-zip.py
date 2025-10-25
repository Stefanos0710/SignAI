from pathlib import Path
import zipfile
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_DIR = PROJECT_ROOT / "app" / "builds" / "zip"

# make zip folder if not there
os.makedirs(ZIP_DIR, exist_ok=True)

def zip_items(exe_path, updater_path, zip_name, version):
    # start make zip file
    print("Starting to create zip file...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # add exe file in exe folder
        if exe_path.exists():
            arcname = f"{version}/exe/{exe_path.name}"
            zipf.write(exe_path, arcname)
            print(f"Do file: {arcname}")
        else:
            print(f"Warning: {exe_path.resolve()} not here or skip.")
        # add updater folder content
        if updater_path.exists() and updater_path.is_dir():
            print(f"Do directory: {updater_path}")
            for file in updater_path.rglob('*'):
                if file.is_file():
                    arcname = f"{version}/updater/{file.relative_to(updater_path)}"
                    zipf.write(file, arcname)
                    print(f"\t{arcname}")
                else:
                    print(f"\tSkip directory: {file.relative_to(updater_path)}")
        else:
            print(f"Warning: {updater_path.resolve()} not here or skip.")
    print(f"Zip file make: {zip_name}")

def get_current_version():
    version_file = PROJECT_ROOT / "app" / "updater" / "version.txt"
    if not version_file.exists():
        print(f"Version file not here: {version_file.resolve()}")
        return "No_version"
    with open(version_file, "r") as f:
        return f.read().strip()

if __name__ == "__main__":
    version = get_current_version()
    exe_path = PROJECT_ROOT / "app" / "dist" / "SignAI - Desktop.exe"
    updater_path = PROJECT_ROOT / "app" / "updater"
    zip_name = ZIP_DIR / f"v{version}.zip"
    zip_items(exe_path, updater_path, zip_name, version)
