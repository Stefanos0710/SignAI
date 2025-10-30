"""

also i should implement this in the final app script


"""

from pathlib import Path
import zipfile
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_DIR = PROJECT_ROOT / "app" / "builds" / "zip"

# make zip folder if not there
os.makedirs(ZIP_DIR, exist_ok=True)

def zip_items(dist_dir, zip_name, version):
    print("Starting to create zip file...")

    desktop_exe = dist_dir / "SignAI - Desktop.exe"
    updater_exe = dist_dir / "SignAI - Updater.exe"

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # add SignAI - Desktop.exe
        if desktop_exe.exists():
            arcname = desktop_exe.name
            zipf.write(desktop_exe, arcname)
            print(f"Do file: {arcname}")
        else:
            print(f"Warning: {desktop_exe.resolve()} not here or skip.")

        # add SignAI - Updater.exe
        if updater_exe.exists():
            arcname = updater_exe.name
            zipf.write(updater_exe, arcname)
            print(f"Do file: {arcname}")
        else:
            print(f"Warning: {updater_exe.resolve()} not here or skip.")

    print(f"Zip file made: {zip_name}")

def get_current_version():
    version_file = PROJECT_ROOT / "app" / "updater" / "version.txt"
    if not version_file.exists():
        print(f"Version file not here: {version_file.resolve()}")
        return "No_version"
    with open(version_file, "r") as f:
        return f.read().strip()

if __name__ == "__main__":
    version = get_current_version()
    dist_dir = PROJECT_ROOT / "app" / "dist"
    zip_name = ZIP_DIR / f"{version}.zip"
    zip_items(dist_dir, zip_name, version)
