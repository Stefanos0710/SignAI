from pathlib import Path
import requests
import os
import shutil
import zipfile
import re
import time
import subprocess
import sys

# Fallback für dotenv, falls nicht installiert
try:
    from dotenv import load_dotenv
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

"""
# Updater Workflow

This document outlines the step-by-step process for updating the application safely while preserving user data.

---

## 1. Check for Updates
- The updater first checks if a new version of the application is available on GitHub.

---

## 2. Backup User Data
- Before making any changes, save important folders into temporary folders:
  - `settings` → `tmp_settings`
  - `videos` → `tmp_videos`
  - `data` → `tmp_data`
- This ensures that no user data is lost during the update process.

---

## 3. Clean Old Application
- Delete all existing application files to prepare for the new version.
- **Important:** Keep critical files such as the uninstaller and other essential files that should not be removed.

---

## 4. Download and Extract New Version
- Locate the appropriate version from GitHub.
- Download the release zip file.
- Extract its contents into the application directory.

---

## 5. Restore User Data
- Move the previously backed-up folders (`settings`, `videos`, `data`) back into the application directory.
- This ensures that all user settings and files remain intact after the update.

---

## 6. Clean Temporary Files
- After restoring user data, delete all temporary folders (`tmp_settings`, `tmp_videos`, `tmp_data`) to keep the application directory clean.


r"C:\Program Files (x86)\SignAI - Desktop"
tmp_folders vs. save_files

"""


class Updater:
    def __init__(self):
        # Load environment variables from a .env file
        load_dotenv()
        self.API_KEY = os.getenv("GITHUB_TOKEN")
        self.OWNER = os.getenv("REPO_OWNER")
        self.REPO = os.getenv("REPO_NAME")

        # make url and headers for api request
        self.API_URL = f"https://api.github.com/repos/{self.OWNER}/{self.REPO}/releases/latest"
        self.HEADERS = {"Authorization": f"token {self.API_KEY}"}

        # files and folders to save during update
        self.save_files = ["settings/", "videos/", "data/"]
        self.dont_delete = ["tmp_data/", "tmp_updater/", "tmp_updater/", "Uninstall SignAI - Desktop_lang.ifl", "Uninstall SignAI - Desktop.exe", "Uninstall SignAI - Desktop.dat"]

        # tmp folders
        self.tmp_folders = ["tmp_data/", "tmp_updater/", "tmp_settings/", "tmp_videos/"]

    def check_for_updates(self, current_version):
        print("Checking for updates...")
        response = requests.get(self.API_URL, headers=self.HEADERS)

        # check id response failed
        if response.status_code != 200:
            print("Failed to fetch release information.")
            return None

        data = response.json()
        latest_tag = data.get("tag_name", "")
        print(f"Latest realese tag: {latest_tag}")

        # check if the names are same
        if latest_tag != current_version:
            # extract numbers and level (alpha beta final)
            def parse_version(v):
                v = v.lower().lstrip('v')
                numbers = tuple(map(int, re.findall(r'\d+', v))) or (0,)
                if "alpha" in v: stage=0
                elif "beta" in v: stage=1
                else: stage=2

                return numbers + (stage,)

            latest_v = parse_version(latest_tag)
            current_v = parse_version(current_version)

            if latest_v > current_v:
                print(f"Update available: {latest_tag}")

                # update the verion.txt file
                self.update_version_file(latest_tag)

                for asset in data.get("assets", []):
                    if asset.get("name", "").endswith(".zip"):
                        return asset["browser_download_url"]

                print("No zip asset found in the latest release.")

            else:
                print("Already on the latest version.")

        else:
            print("Already on the latest version.")

        return None

    def current_version(self):
        version_file = Path("version.txt")
        if version_file.exists():
            with open(version_file, "r") as f:
                return f.read().strip()
        else:
            print("No version file.")
            return "0.0.0"

    def get_project_paths(self):
        if getattr(sys, 'frozen', False):
            return Path(sys.executable).parent
        else:
            return Path(__file__).resolve().parent.parent

    def download_new_version(self, download_url):
        app_dir = self.get_project_paths()
        tmp_version_dir = app_dir / "tmp_version"
        update_zip_path = tmp_version_dir / "update.zip"

        # create or clean update directory (tmp_version)
        if tmp_version_dir.exists():
            shutil.rmtree(tmp_version_dir)
        tmp_version_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading the newest version: {update_zip_path}")
        with requests.get(download_url, headers=self.HEADERS, stream=True) as response:
            if response.status_code != 200:
                print("Failed to download the new version.")
                return None
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(update_zip_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        print(f"\rProgress: {downloaded / total:.0%}", end="")
        print("\nDownload finished!")
        return update_zip_path

    def create_tmp_data(self):
        """Backup settings, videos, and data into their respective tmp folders."""
        # app_dir = self.get_project_paths()
        app_dir = self.get_project_paths()
        backup_map = {
            "settings": "tmp_settings",
            "videos": "tmp_videos",
            "data": "tmp_data"
        }
        for src, tmp in backup_map.items():
            src_path = app_dir / src
            tmp_path = app_dir / tmp
            if src_path.exists():
                if tmp_path.exists():
                    shutil.rmtree(tmp_path)
                shutil.copytree(src_path, tmp_path)
                print(f"Backed up {src} to {tmp}")
            else:
                print(f"Source folder {src} not found, skipping backup.")
        print("User data backup to tmp folders completed.")

    def delete_old_data(self):
        # get the app dir from func get_project_paths
        app_dir = self.get_project_paths()

        deleted_count = 0
        skipped_count = 0

        for item in app_dir.iterdir():
            if item.name in self.dont_delete:
                print(f"Skipped: {item.name}")
                skipped_count += 1
                continue
            try:
                if item.is_file():
                    item.unlink()
                    print(f"Deleted file: {item.name}")

                else:
                    shutil.rmtree(item)
                    print(f"Deleted directory: {item.name}")
                deleted_count += 1

            except Exception as e:
                print(f"Failed to delete {item.name}: {e}")

        print(f"\n Cleanup complete! Deleted: {deleted_count}, Skipped: {skipped_count}")

    def update_version_file(self, new_version):
        version_file = Path("version.txt")
        try:
            with open(version_file, "w") as f:
                f.write(new_version)
            print(f"Updated version file to: {new_version}")
        except Exception as e:
            print(f"Failed to update version file: {e}")

    def unzip_new_version(self, zip_path):
        zip_path = Path(zip_path)
        extract_to = zip_path.parent # in the same dic as zip file

        extract_to.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        # del zip file
        zip_path.unlink()

        print(f"Unzipped '{zip_path.name}' into '{extract_to}'")

    def setup_new_version(self, new_version):
        new_version = Path(new_version)
        app_dir = self.get_project_paths()

        for item in new_version.iterdir():  # gets throuh all files and folder in the tmp_version
            target = new_version / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)  # copy dictory with all content
            else:
                shutil.copy2(item, target)  # copy file with all content

    def delete_tmp_files(self):
        app_dir = self.get_project_paths()

        for folder in self.tmp_folders:
            tmp_path = app_dir / folder
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
                print(f"Deleted temporary folder: {folder}")
            else:
                print(f"Temporary folder not found, skipped: {folder}")

    def get_tmp_data(self):
        app_dir = self.get_project_paths()

        tmp_folders = {
            "tmp_settings": "settings",
            "tmp_videos": "videos",
            "tmp_data": "data"
        }

        for tmp_name, target_name in tmp_folders.items():
            source = app_dir / tmp_name
            target = app_dir / target_name

            if source.exists():
                # Ordner im App-Verzeichnis vorher löschen, falls er existiert
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
                print(f"Restored '{target_name}' from '{tmp_name}'")
            else:
                print(f"Backup folder '{tmp_name}' not found, skipping.")

    def start(self):
        current_version = self.current_version()
        download_url = self.check_for_updates(current_version)
        if not download_url:
            print("No update needed.")
            return

        # 1. Backup user data
        self.create_tmp_data()

        # 2. Delete old app files
        self.delete_old_data()

        # 3. Download new version
        zip_path = self.download_new_version(download_url)
        if not zip_path:
            print("Failed to download new version.")
            return

        # 4. Unzip new version
        self.unzip_new_version(zip_path)

        # 5. Restore user data
        self.get_tmp_data()

        # 6. Delete temporary files
        self.delete_tmp_files()

        print("Update completed successfully!")

        # 7. Restart app
        time.sleep(2)
        app_dir = self.get_project_paths()
        exe_path = app_dir / "SignAI - Desktop.exe"
        if exe_path.exists():
            subprocess.Popen([str(exe_path)])
            print(f"Started app: {exe_path}")
        else:
            print(f"App-EXE not found: {exe_path}")

        # 8. Close updater
        time.sleep(2)
        subprocess.run(["taskkill", "/f", "/im", "SignAI - Updater.exe"])

# for testing, del in production
if __name__ == '__main__':
    updater = Updater()
    updater.start()
