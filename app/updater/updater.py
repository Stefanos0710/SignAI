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

        # tmp folders
        self.tmp_folders = ["tmp_data/", "tmp_updater/", "tmp_version/"]

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
        if self.check_for_updates(self.current_version()):
            pass

# for testing, del in production
if __name__ == '__main__':
    u = Updater()
    download_url = u.check_for_updates(u.current_version())
    if download_url:
        u.download_new_version(download_url)
