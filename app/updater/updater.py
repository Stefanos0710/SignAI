from pathlib import Path
import requests
from dotenv import load_dotenv
import os
import shutil
import zipfile
import re

"""
TODOs:
# [ ] make that the new verion is written down after that

Roadmap:    
- backup user data (settings, videos, data)
- get name/ url / version of the latest release from github
- download zip in tmp folder
- unzip in tmp folder
- replace existing files with new ones
- restore user data


C:\Program Files (x86)\SignAI - Desktop


"""

# class Updater:
#     def __init__(self):
#         # Load environment variables from a .env file
#         load_dotenv()
#         self.API_KEY = os.getenv("GITHUB_TOKEN")
#         self.OWNER = os.getenv("REPO_OWNER")
#         self.REPO = os.getenv("REPO_NAME")
#
#         # make url and headers for api request
#         self.API_URL = f"https://api.github.com/repos/{self.OWNER}/{self.REPO}/releases/latest"
#         self.HEADERS = {"Authorization": f"token {self.API_KEY}"}
#
#         # files and folders to save during update
#         self.save_files = ["settings", "videos/", "data/"]
#
#         # main paths
#         self.USER_HOME = Path.home()
#         self.BACKUP_DIR = self.USER_HOME / "AppData" / "Local" / "SignAI" / "backup"
#         self.UPDATE_DIR = self.USER_HOME / "AppData" / "Local" / "SignAI" / "update_tmp"
#         self.UPDATE_FILE = self.UPDATE_DIR / "update.zip"
#
#     def start(self):
#         """ Here is the logic to start the update process """
#         print("-- Staring update --")
#
#         # save user data
#         self.save_user_data()
#         print("User data saved.")
#
#         # get current version and print it
#         current_version = self.get_current_version()
#         print(f"Current version: {current_version}")
#
#         # check for updates
#         download_url, latest_version = self.check_for_updates(current_version)
#         if not download_url:
#             print("No updates available.")
#             return
#         print(f"Update available: {latest_version}")
#
#         zip_path = self.load_new_version(download_url)
#         if not zip_path:
#             print("Failed to download the update.")
#             return
#
#         self.unzip_new_version()
#         self.replace_old_version()
#         self.get_user_data()
#
#         print("-- Finished update! --")
#
#     def get_current_version(self):
#         """Get the local version from version.txt file"""
#         version_file = Path("version.txt")
#         if version_file.exists():
#             with open(version_file, "r") as f:
#                 return f.read().strip()
#         else:
#             print("No version file.")
#             return "0.0.0"
#
#     def check_for_updates(self, current_version):
#         """Check for updates by comparing the current version with the latest release on GitHub and get the download URL if an update is available."""
#         response = requests.get(self.API_URL, headers=self.HEADERS)
#         print("Checking for updates...")
#
#         # check id response failed
#         if response.status_code != 200:
#             print("Failed to fetch release information.")
#             return None
#
#         data = response.json()
#         assets = data.get("assets", [])
#
#         # check if there are assets in the release
#         if not assets:
#             print("No assets found in the latest release.")
#             return None
#
#         for asset in assets:
#             name = asset.get("name", "")
#             if name.endswith(".zip"):
#                 print(f"Found a zip asset: {name}")
#                 return asset["browser_download_url"]
#
#         print("No assets found in the latest release.")
#         return None
#
#     def load_new_version(self, download_url):
#         """Download the new version from the provided download URL and save it in SignAI/update_tmp."""
#         USER_HOME = Path.home()
#         UPDATE_DIR = USER_HOME / "AppData" / "Local" / "SignAI" / "update_tmp"
#         UPDATE_FILE = UPDATE_DIR / "update.zip"
#
#         # create or clean update directory
#         if UPDATE_DIR.exists():
#             shutil.rmtree(UPDATE_DIR)
#         UPDATE_DIR.mkdir(parents=True, exist_ok=True)
#
#         print("Starting download of the new version...")
#
#         # download the file
#         with requests.get(download_url, headers=self.HEADERS, stream=True) as response:
#             if response.status_code != 200:
#                 print("Failed to download update file.")
#                 return None
#
#             total = int(response.headers.get("content-length", 0))
#             downloaded = 0
#
#             with open(UPDATE_FILE, "wb") as f:
#                 for chunk in response.iter_content(1024):
#                     if chunk:
#                         f.write(chunk)
#                         downloaded += len(chunk)
#                         # show progress in terminal - should may make later a progress bar in the GUI
#                         print(f"\rProgress: {downloaded / total:.0%}", end="")
#
#         print("\n Download complete!")
#         return UPDATE_FILE
#
#     def unzip_new_version(self):
#         """Unzip the downloaded update file into the update_tmp directory."""
#         print("Unzipping the new version...")
#         try:
#             with zipfile.ZipFile(self.UPDATE_FILE, 'r') as zip_ref:
#                 zip_ref.extractall(self.UPDATE_DIR)
#             print("Unzipping complete!")
#         except zipfile.BadZipFile as e:
#             print(f"Failed to unzip the update file: {e}")
#
#     def replace_old_version(self):
#         """Replace the old version files with the new version files."""
#         print("replacing ikd version files...")
#         try:
#             app_dir = Path.cwd()
#
#             for item in self.UPDATE_DIR.iterdir():
#                 target_path = app_dir / item.name
#                 if target_path.exists():
#                     if target_path.is_file():
#                         target_path.unlink()
#                     elif target_path.is_dir():
#                         shutil.rmtree(target_path)
#
#                 if item.is_file():
#                     shutil.copy2(item, target_path)
#                 elif item.is_dir():
#                     shutil.copytree(item, target_path)
#                 print(f"Replaced: {item.name}")
#             print("Replacement complete!")
#         except Exception as e:
#             print(f"Failed to replace old version files: {e}")
#
#     def save_user_data(self):
#         """Save users data before updating.
#         self.save_files = ["settings", "videos/", "data/"]
#         """
#
#         USER_HOME = Path.home()
#         BACKUP_DIR = USER_HOME / "AppData" / "Local" / "SignAI" / "backup"
#
#         # check if backup directory exists, if not create it
#         if BACKUP_DIR.exists():
#             shutil.rmtree(BACKUP_DIR) # delete existing backup
#         BACKUP_DIR.mkdir(parents=True, exist_ok=True)
#
#         for item in self.save_files:
#             if os.path.exists(item):
#                 source_path = Path(item)
#                 destination_path = BACKUP_DIR / source_path.name
#                 if source_path.is_file():
#                     shutil.copy2(source_path, destination_path)
#                 elif source_path.is_dir():
#                     shutil.copytree(source_path, destination_path)
#                 print(f"Saved: {item}")
#             else:
#                 print(f"{item} does not exist, skipping backup.")
#
#     def get_user_data(self):
#         """Retrieve users data after updating.
#         self.save_files = ["settings", "videos/", "data/"]
#         """
#
#         USER_HOME = Path.home()
#         BACKUP_DIR = USER_HOME / "AppData" / "Local" / "SignAI" / "backup"
#
#         if not BACKUP_DIR.exists():
#             """ Implement here logic to create "fake backup" or just handle it """
#             print("No backups found!")
#
#         for item in self.save_files:
#             backup_item = BACKUP_DIR / Path(item).name
#             if backup_item.exists():
#                 if backup_item.is_file():
#                     shutil.copy2(backup_item, item)
#                 elif backup_item.is_dir():
#                     if os.path.exists(item):
#                         shutil.rmtree(item)
#                     shutil.copytree(backup_item, item)
#                 print(f"{item} restored successfully.")


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
        self.dont_delete = ["tmp_data/", "tmp_updater/", "Uninstall SignAI - Desktop_lang.ifl", "Uninstall SignAI - Desktop.exe", "Uninstall SignAI - Desktop.dat"]

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
        # get path to app folder
        current_file = Path(__file__).resolve()
        updater_dir = current_file.parent
        app_dir = updater_dir.parent

        # log the paths
        print(f"Updater dir: {updater_dir}")
        print(f"App dir: {app_dir}")

        return app_dir

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

    def create_tmp_updater(self):
        pass

    def create_tmp_data(self):
        # get the app dir from func get_project_paths
        app_dir = self.get_project_paths()

        # get the tmp data dir
        tmp_data_dir = self.get_project_paths() / "tmp_data"

        # create the tmp data dir
        os.makedirs(tmp_data_dir, exist_ok=True)

        # copy all save_files into tmp_data
        for folder in self.save_files:
            source_path = app_dir / folder
            target_path = tmp_data_dir / folder
            if source_path.exists():
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)
                print(f"Copied {folder} -> tmp_data/{folder}")
            else:
                print(f"Folder {folder} not found, skipped.")

        print("User data saved to tmp_data.")

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

    def start(self):
        if self.check_for_updates(self.current_version()):
            pass

# for testing, del in production
if __name__ == '__main__':
    u = Updater()
    download_url = u.check_for_updates(u.current_version())
    if download_url:
        u.download_new_version(download_url)