from pathlib import Path
import requests
import os
import shutil
import zipfile
import re
import time
import subprocess
import sys
import logging

# Fallback für dotenv, falls nicht installiert
try:
    from dotenv import load_dotenv
except ImportError:
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


def get_project_paths():
    """Return base app directory.

    Priority:
    1) SIGN_AI_APP_DIR (env)
    2) If frozen (PyInstaller): parent of sys.executable (onedir)
    3) Dev mode: parent of this file (../app)
    """
    app_dir_env = os.environ.get("SIGN_AI_APP_DIR")

    if app_dir_env and Path(app_dir_env).exists():
        base_dir = Path(app_dir_env)
        print(f"DEBUG get_project_paths: Using env var path: {base_dir}")
    else:
        if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
            base_dir = Path(sys.executable).parent
        else:
            base_dir = Path(__file__).resolve().parent.parent
        print(f"DEBUG get_project_paths: Fallback path used: {base_dir}")

    return base_dir


class Updater:
    def __init__(self):
        # Determine base dir early so other methods use the correct path
        self.base_dir = get_project_paths()
        print(f"Updater base dir: {self.base_dir}")

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
        self.dont_delete = ["tmp_videos/", "tmp_updater/", "tmp_updater/", "Uninstall SignAI - Desktop_lang.ifl", "Uninstall SignAI - Desktop.exe", "Uninstall SignAI - Desktop.dat"]

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

                # Find and return the download URL for the zip file
                assets = data.get("assets", [])
                print(f"Found {len(assets)} assets in release")

                for asset in assets:
                    asset_name = asset.get("name", "")
                    print(f"  - Asset: {asset_name}")
                    if asset_name.endswith(".zip"):
                        download_url = asset["browser_download_url"]
                        print(f"[OK] Selected ZIP asset: {asset_name}")
                        print(f"  Download URL: {download_url}")
                        return download_url

                print("[WARNING] No zip asset found in the latest release.")

            else:
                print("Already on the latest version.")

        else:
            print("Already on the latest version.")

        return None

    def current_version(self):
        version_file = self.base_dir / "version.txt"
        if version_file.exists():
            with open(version_file, "r") as f:
                version = f.read().strip()
                print(f"Current version from file: {version}")
                return version
        else:
            print(f"No version file found at {version_file}")
            return "0.0.0"

    def download_new_version(self, download_url):
        app_dir = self.base_dir
        tmp_version_dir = app_dir / "tmp_version"
        update_zip_path = tmp_version_dir / "update.zip"

        # create or clean update directory (tmp_version)
        if tmp_version_dir.exists():
            shutil.rmtree(tmp_version_dir)
        tmp_version_dir.mkdir(parents=True, exist_ok=True)

        print(f"Download URL: {download_url}")
        print(f"Downloading to: {update_zip_path}")

        try:
            # For private releases, we need to use the GitHub API with authentication
            # Instead of browser_download_url, use the API endpoint
            if "github.com" in download_url and self.API_KEY:
                # Extract owner, repo, tag, and filename from the URL
                # URL format: https://github.com/owner/repo/releases/download/tag/filename
                parts = download_url.split('/')
                owner = parts[3]
                repo = parts[4]
                tag = parts[7]
                filename = parts[8]

                # Get the asset ID from the API
                release_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
                release_response = requests.get(release_url, headers=self.HEADERS)

                if release_response.status_code == 200:
                    release_data = release_response.json()
                    assets = release_data.get("assets", [])

                    for asset in assets:
                        if asset["name"] == filename:
                            # Use the API URL for downloading with authentication
                            api_download_url = asset["url"]
                            print(f"Using API download URL: {api_download_url}")

                            # Download using API with Accept header for asset download
                            headers = self.HEADERS.copy()
                            headers["Accept"] = "application/octet-stream"

                            with requests.get(api_download_url, headers=headers, stream=True, allow_redirects=True) as response:
                                print(f"Response status: {response.status_code}")

                                if response.status_code != 200:
                                    print(f"Failed to download. Status code: {response.status_code}")
                                    print(f"Response headers: {dict(response.headers)}")
                                    return None

                                total = int(response.headers.get("content-length", 0))
                                downloaded = 0

                                with open(update_zip_path, "wb") as f:
                                    for chunk in response.iter_content(chunk_size=8192):
                                        if chunk:
                                            f.write(chunk)
                                            downloaded += len(chunk)
                                            if total > 0:
                                                print(f"\rProgress: {downloaded / total:.0%}", end="")
                                            else:
                                                print(f"\rDownloaded: {downloaded / 1024 / 1024:.2f} MB", end="")

                                print(f"\nDownload finished! File size: {downloaded / 1024 / 1024:.2f} MB")

                                # Verify the file exists and has content
                                if not update_zip_path.exists():
                                    print("ERROR: Downloaded file does not exist!")
                                    return None

                                file_size = update_zip_path.stat().st_size
                                if file_size == 0:
                                    print("ERROR: Downloaded file is empty!")
                                    return None

                                print(f"Verified: File exists with size {file_size / 1024 / 1024:.2f} MB")
                                return update_zip_path

                    print("ERROR: Asset not found in release!")
                    return None

            # Fallback to direct download (for public releases)
            with requests.get(download_url, stream=True, allow_redirects=True) as response:
                print(f"Response status: {response.status_code}")

                if response.status_code != 200:
                    print(f"Failed to download. Status code: {response.status_code}")
                    print(f"Response headers: {dict(response.headers)}")
                    print(f"Response text: {response.text[:500]}")
                    return None

                total = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(update_zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                print(f"\rProgress: {downloaded / total:.0%}", end="")
                            else:
                                print(f"\rDownloaded: {downloaded / 1024 / 1024:.2f} MB", end="")

                print(f"\nDownload finished! File size: {downloaded / 1024 / 1024:.2f} MB")

                # Verify the file exists and has content
                if not update_zip_path.exists():
                    print("ERROR: Downloaded file does not exist!")
                    return None

                file_size = update_zip_path.stat().st_size
                if file_size == 0:
                    print("ERROR: Downloaded file is empty!")
                    return None

                print(f"Verified: File exists with size {file_size / 1024 / 1024:.2f} MB")
                return update_zip_path

        except Exception as e:
            print(f"Download exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_tmp_data(self):
        """Backup settings, videos, and data into their respective tmp folders."""
        app_dir = self.base_dir
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
        app_dir = self.base_dir

        deleted_count = 0
        skipped_count = 0

        # dont_delete list
        dont_delete_names = [
            "tmp_videos", "tmp_version", "tmp_data", "tmp_settings",
            "SignAI - Updater.exe",  # Keep the updater EXE (it's in the same folder)
            "updater",  # Keep the updater source folder (if it exists)
            "Uninstall SignAI - Desktop_lang.ifl",
            "Uninstall SignAI - Desktop.exe",
            "Uninstall SignAI - Desktop.dat",
            "version.txt",  # Keep version file
            ".env",  # Keep environment variables
            "_internal"  # Do not delete PyInstaller internal folder
        ]

        for item in app_dir.iterdir():
            if item.name in dont_delete_names:
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
        version_file = self.base_dir / "version.txt"
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

    def setup_new_version(self, tmp_version_dir):
        """Move all files from tmp_version to app directory."""
        tmp_version_dir = Path(tmp_version_dir)
        app_dir = self.base_dir

        print(f"Moving files from {tmp_version_dir} to {app_dir}")

        for item in tmp_version_dir.iterdir():
            target = app_dir / item.name

            # Skip if it's a folder we want to preserve (will be restored later)
            if item.name in ["settings", "videos", "data"]:
                print(f"Skipped '{item.name}' (will be restored from backup)")
                continue

            try:
                if item.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(item, target)
                    print(f"Copied directory: {item.name}")
                else:
                    shutil.copy2(item, target)
                    print(f"Copied file: {item.name}")
            except Exception as e:
                print(f"Failed to copy {item.name}: {e}")

    def delete_tmp_files(self):
        app_dir = self.base_dir

        for folder in self.tmp_folders:
            tmp_path = app_dir / folder
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
                print(f"Deleted temporary folder: {folder}")
            else:
                print(f"Temporary folder not found, skipped: {folder}")

    def get_tmp_data(self):
        app_dir = self.base_dir

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
        print("\n=== STEP 1: BACKUP USER DATA ===")
        self.create_tmp_data()

        # 2. Delete old app files
        print("\n=== STEP 2: DELETE OLD FILES ===")
        self.delete_old_data()

        # 3. Download new version
        print("\n=== STEP 3: DOWNLOAD NEW VERSION ===")
        zip_path = self.download_new_version(download_url)
        # if not zip_path:
        #     raise RuntimeError("Failed to download new version.")

        # 4. Extract new version
        print("\n=== STEP 4: EXTRACT NEW VERSION ===")
        self.unzip_new_version(zip_path)

        # 5. Move new files to app directory
        print("\n=== STEP 5: SETUP NEW VERSION ===")
        tmp_version_dir = self.base_dir / "tmp_version"
        self.setup_new_version(tmp_version_dir)

        # 6. Restore user data
        print("\n=== STEP 6: RESTORE USER DATA ===")
        self.get_tmp_data()

        # 7. Verify that main exe exists before cleanup
        exe_path = self.base_dir / "SignAI - Desktop.exe"
        if not exe_path.exists():
            raise FileNotFoundError("Main application EXE not found after update!")

        # 8. Cleanup tmp folders AFTER everything is successful
        print("\n=== STEP 7: CLEANUP TEMPORARY FILES ===")
        self.delete_tmp_files()

        # Also delete tmp_version folder
        tmp_version_dir = self.base_dir / "tmp_version"
        if tmp_version_dir.exists():
            shutil.rmtree(tmp_version_dir)
            print("Deleted tmp_version folder")

        print("\n[SUCCESS] Update completed successfully!")

        # 9. Restart app
        time.sleep(2)
        subprocess.Popen([str(exe_path)])
        print(f"Started app: {exe_path}")

        # Kill updater after short delay
        time.sleep(2)
        subprocess.run(["taskkill", "/f", "/im", "SignAI - Updater.exe"])


# for testing, del in production
if __name__ == '__main__':
    updater = Updater()
    updater.start()
