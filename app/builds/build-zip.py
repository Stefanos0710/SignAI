from pathlib import Path
import zipfile
import os
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_DIR = PROJECT_ROOT / "app" / "builds" / "zip"

# make zip folder if not there
os.makedirs(ZIP_DIR, exist_ok=True)

def zip_items(desktop_dir, updater_dir, zip_name, version):
    """Create release zip with desktop and updater folders.

    Args:
        desktop_dir: Path to SignAI - Desktop folder (onedir build)
        updater_dir: Path to SignAI - Updater folder (onedir build)
        zip_name: Output zip file path
        version: Version string for logging
    """
    print(f"\nStarting to create zip file v{version}...")
    print(f"Desktop dir: {desktop_dir}")
    print(f"Updater dir: {updater_dir}")
    print(f"Output zip: {zip_name}")

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add desktop folder content
        if desktop_dir.exists() and desktop_dir.is_dir():
            print(f"\nAdding Desktop folder: {desktop_dir}")
            desktop_base = "SignAI - Desktop"
            for file in desktop_dir.rglob('*'):
                if "__pycache__" in file.parts:
                    continue
                if file.is_file():
                    # Keep folder structure but use SignAI - Desktop as root
                    arcname = os.path.join(desktop_base, file.relative_to(desktop_dir))
                    zipf.write(file, arcname)
                    print(f"  + {arcname}")
        else:
            print(f"Warning: Desktop folder not found: {desktop_dir}")

        # Add updater folder content
        if updater_dir.exists() and updater_dir.is_dir():
            print(f"\nAdding Updater folder: {updater_dir}")
            updater_base = "SignAI - Updater"
            for file in updater_dir.rglob('*'):
                if "__pycache__" in file.parts:
                    continue
                if file.is_file():
                    # Keep folder structure but use SignAI - Updater as root
                    arcname = os.path.join(updater_base, file.relative_to(updater_dir))
                    zipf.write(file, arcname)
                    print(f"  + {arcname}")
        else:
            print(f"Warning: Updater folder not found: {updater_dir}")

    size_mb = Path(zip_name).stat().st_size / (1024*1024)
    print(f"\nZip file created: {zip_name}")
    print(f"Size: {size_mb:.1f} MB")

if __name__ == "__main__":
    print("=== SignAI Release Zip Builder ===\n")
    print("PLease enter a valid version (e.g. v1.2.4 or v0.2.3-alpha):")
    version = input("=> ")

    # Use onedir builds from dist/
    desktop_dir = PROJECT_ROOT / "app" / "dist" / "SignAI - Desktop"
    updater_dir = PROJECT_ROOT / "app" / "dist" / "SignAI - Updater"

    zip_name = ZIP_DIR / f"{version}.zip"

    zip_items(desktop_dir, updater_dir, zip_name, version)
