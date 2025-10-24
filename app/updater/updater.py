from pathlib import Path
import requests
from dotenv import load_dotenv
import os
import shutil
import zipfile

"""
TODOs:


Roadmap:    
- backup user data (settings, videos, data)
- get name/ url / version of the latest release from github
- download zip in tmp folder
- unzip in tmp folder
- replace existing files with new ones
- restore user data

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
        self.save_files = ["settings", "videos/", "data/"]

        # main paths
        self.USER_HOME = Path.home()
        self.BACKUP_DIR = self.USER_HOME / "AppData" / "Local" / "SignAI" / "backup"
        self.UPDATE_DIR = self.USER_HOME / "AppData" / "Local" / "SignAI" / "update_tmp"
        self.UPDATE_FILE = self.UPDATE_DIR / "update.zip"

    def start(self):
        """ Here is the logic to start the update process """
        pass

    def check_for_updates(self, current_version):
        """Check for updates by comparing the current version with the latest release on GitHub and get the download URL if an update is available."""
        response = requests.get(self.API_URL, headers=self.HEADERS)
        print("Checking for updates...")

        # check id response failed
        if response.status_code != 200:
            print("Failed to fetch release information.")
            return None

        data = response.json()
        assets = data.get("assets", [])

        # check if there are assets in the release
        if not assets:
            print("No assets found in the latest release.")
            return None

        for asset in assets:
            name = asset.get("name", "")
            if name.endswith(".zip"):
                print(f"Found a zip asset: {name}")
                return asset["browser_download_url"]

        print("No assets found in the latest release.")
        return None

    def load_new_version(self, download_url):
        """Download the new version from the provided download URL and save it in SignAI/update_tmp."""
        USER_HOME = Path.home()
        UPDATE_DIR = USER_HOME / "AppData" / "Local" / "SignAI" / "update_tmp"
        UPDATE_FILE = UPDATE_DIR / "update.zip"

        # create or clean update directory
        if UPDATE_DIR.exists():
            shutil.rmtree(UPDATE_DIR)
        UPDATE_DIR.mkdir(parents=True, exist_ok=True)

        print("Starting download of the new version...")

        # download the file
        with requests.get(download_url, headers=self.HEADERS, stream=True) as response:
            if response.status_code != 200:
                print("Failed to download update file.")
                return None

            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(UPDATE_FILE, "wb") as f:
                for chunk in response.iter_content(1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # show progress in terminal - should may make later a progress bar in the GUI
                        print(f"\rProgress: {downloaded / total:.0%}", end="")

        print("\n Download complete!")
        return UPDATE_FILE

    def unzip_new_version(self):
        """Unzip the downloaded update file into the update_tmp directory."""
        print("Unzipping the new version...")
        try:
            with zipfile.ZipFile(self.UPDATE_FILE, 'r') as zip_ref:
                zip_ref.extractall(self.UPDATE_DIR)
            print("Unzipping complete!")
        except zipfile.BadZipFile as e:
            print(f"Failed to unzip the update file: {e}")

    def replace_old_version(self):
        """Replace the old version files with the new version files."""
        print("Replacing the old version files with new ones...")


    def save_user_data(self):
        """Save users data before updating.
        self.save_files = ["settings", "videos/", "data/"]
        """

        USER_HOME = Path.home()
        BACKUP_DIR = USER_HOME / "AppData" / "Local" / "SignAI" / "backup"

        # check if backup directory exists, if not create it
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR) # delete existing backup
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        for item in self.save_files:
            if os.path.exists(item):
                source_path = Path(item)
                destination_path = BACKUP_DIR / source_path.name
                if source_path.is_file():
                    shutil.copy2(source_path, destination_path)
                elif source_path.is_dir():
                    shutil.copytree(source_path, destination_path)
                print(f"Saved: {item}")
            else:
                print(f"{item} does not exist, skipping backup.")

    def get_user_data(self):
        """Retrieve users data after updating.
        self.save_files = ["settings", "videos/", "data/"]
        """

        USER_HOME = Path.home()
        BACKUP_DIR = USER_HOME / "AppData" / "Local" / "SignAI" / "backup"

        if not BACKUP_DIR.exists():
            """ Implement here logic to create "fake backup" or just handle it """
            print("No backups found!")

        for item in self.save_files:
            backup_item = BACKUP_DIR / Path(item).name
            if backup_item.exists():
                if backup_item.is_file():
                    shutil.copy2(backup_item, item)
                elif backup_item.is_dir():
                    if os.path.exists(item):
                        shutil.rmtree(item)
                    shutil.copytree(backup_item, item)
                print(f"{item} restored successfully.")
