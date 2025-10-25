from pathlib import Path
import zipfile

def zip_folders(folder_path, zip_name):
    folder_path = Path(folder_path)

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in folder_path.rglob('*'):
            zipf.write(file, file.relative_to(folder_path))
    print(f"Created zip file: {zip_name}")


def get_current_version():
    version_file = Path("version.txt")
    if version_file.exists():
        with open(version_file, "r") as f:
            return f.read().strip()
    else:
        print("No version file.")
        return "No version"

# folders and files to include in the zip
files = [
    "settings",
    "videos/",
    "data/"
]

if __name__ == "__main__":
    zip_name = f"{get_current_version()}.zip"
    folder_path = "/zip"

    # create zip
    zip_folders(folder_path, zip_name)
