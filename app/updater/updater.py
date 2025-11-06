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

# Fallback for dotenv, install if not available
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

    Extra: If running from the updater folder, try to locate a sibling 'SignAI - Desktop' folder
    or any folder in the parent that contains 'SignAI - Desktop.exe'.
    """
    app_dir_env = os.environ.get("SIGN_AI_APP_DIR")

    if app_dir_env and Path(app_dir_env).exists():
        base_dir = Path(app_dir_env)
        print(f"DEBUG get_project_paths: Using env var path: {base_dir}")
        return base_dir

    # Default base (exe dir when frozen; repo app/ in dev)
    if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
        exe_dir = Path(sys.executable).parent
    else:
        exe_dir = Path(__file__).resolve().parent.parent

    # If desktop exe is here, use it
    if (exe_dir / "SignAI - Desktop.exe").exists():
        print(f"DEBUG get_project_paths: Fallback path used (has desktop exe): {exe_dir}")
        return exe_dir

    # If we're inside an updater folder, try to find sibling desktop folder
    parent = exe_dir.parent
    candidate = parent / "SignAI - Desktop"
    if candidate.exists() and (candidate / "SignAI - Desktop.exe").exists():
        print(f"DEBUG get_project_paths: Found sibling desktop folder: {candidate}")
        return candidate

    # Scan parent for any folder containing the desktop exe
    try:
        for child in parent.iterdir():
            if child.is_dir() and (child / "SignAI - Desktop.exe").exists():
                print(f"DEBUG get_project_paths: Found desktop exe in sibling: {child}")
                return child
    except Exception:
        pass

    print(f"DEBUG get_project_paths: Fallback path used: {exe_dir}")
    return exe_dir


class Updater:
    def __init__(self):
        # Determine base_dir early so other methods can use the correct path
        self.base_dir = get_project_paths()
        # Set parent_dir as parent directory of base_dir
        self.parent_dir = self.base_dir.parent
        print(f"Updater base dir: {self.base_dir}")
        print(f"Parent dir: {self.parent_dir}")

        # Load environment variables from .env file
        load_dotenv()

        # Optional configuration file for repository
        repo_cfg = None
        for cfg in [self.base_dir / "repo.json", self.base_dir / "updater" / "repo.json", Path(__file__).resolve().parent / "repo.json"]:
            try:
                if cfg.exists():
                    import json
                    with open(cfg, "r", encoding="utf-8") as f:
                        repo_cfg = json.load(f)
                        print(f"Using repo config file: {cfg}")
                        break
            except Exception:
                pass

        self.API_KEY = os.getenv("GITHUB_TOKEN")
        # Prefer env, then repo.json, then sensible default values based on project link
        self.OWNER = os.getenv("REPO_OWNER") or (repo_cfg.get("owner") if repo_cfg else None) or "Stefanos0710"
        self.REPO = os.getenv("REPO_NAME") or (repo_cfg.get("repo") if repo_cfg else None) or "SignAI"
        if not os.getenv("REPO_OWNER") or not os.getenv("REPO_NAME"):
            print(f"Using fallback repo: {self.OWNER}/{self.REPO}")

        # Create URL and headers for API request
        self.API_URL = (
            f"https://api.github.com/repos/{self.OWNER}/{self.REPO}/releases/latest"
            if self.OWNER and self.REPO else None
        )
        self.HEADERS = {"Accept": "application/vnd.github+json"}
        if self.API_KEY:
            self.HEADERS["Authorization"] = f"token {self.API_KEY}"

        # Files and folders to be backed up during update
        self.save_files = ["settings/", "videos/", "data/"]
        self.dont_delete = ["Uninstall SignAI - Desktop_lang.ifl", "Uninstall SignAI - Desktop.exe", "Uninstall SignAI - Desktop.dat"]

        # Temporary folders at parent level
        self.tmp_folders = ["tmp_data/", "tmp_settings/", "tmp_videos/"]

    def _get_latest_release(self):
        """Fetch latest release JSON or return None."""
        if not self.API_URL:
            print("Repository owner/name not configured; skip update check.")
            return None
        try:
            # Test repository access first
            repo_url = f"https://api.github.com/repos/{self.OWNER}/{self.REPO}"
            repo_resp = requests.get(repo_url, headers=self.HEADERS, timeout=15)
            if repo_resp.status_code == 404:
                print(f"Repository not found: {self.OWNER}/{self.REPO}")
                print("Check REPO_OWNER and REPO_NAME in .env")
                return None
            elif repo_resp.status_code == 401:
                print("GitHub API authentication failed (401).")
                print("Check if GITHUB_TOKEN in .env is valid.")
                return None
            elif repo_resp.status_code != 200:
                print(f"GitHub API error: HTTP {repo_resp.status_code}")
                try:
                    error = repo_resp.json()
                    print(f"API message: {error.get('message', 'No message')}")
                except Exception:
                    print(f"Raw response: {repo_resp.text[:500]}")
                return None

            # Repository exists, try to get latest release
            resp = requests.get(self.API_URL, headers=self.HEADERS, timeout=15)
            if resp.status_code == 404:
                # Fall back to list of releases (includes prereleases)
                url = f"https://api.github.com/repos/{self.OWNER}/{self.REPO}/releases?per_page=1"
                alt = requests.get(url, headers=self.HEADERS, timeout=15)
                if alt.status_code == 200:
                    releases = alt.json() or []
                    if releases:
                        print("Using latest release from list (prerelease allowed).")
                        return releases[0]
                    print("Repository exists but has no releases.")
                    print("Please create a release with a .zip asset first.")
                    return None
                else:
                    print(f"Failed to fetch releases list (HTTP {alt.status_code})")
                    try:
                        error = alt.json()
                        print(f"API message: {error.get('message', 'No message')}")
                    except Exception:
                        print(f"Raw response: {alt.text[:500]}")
                    return None
            if resp.status_code != 200:
                print(f"Failed to fetch latest release (HTTP {resp.status_code})")
                try:
                    error = resp.json()
                    print(f"API message: {error.get('message', 'No message')}")
                except Exception:
                    print(f"Raw response: {resp.text[:500]}")
                return None
            return resp.json()
        except Exception as e:
            print(f"Failed to fetch latest release: {e}")
            return None

    def get_latest_zip_download(self):
        """Return (download_url, tag_name) for the first .zip asset in latest release, or (None, None)."""
        data = self._get_latest_release()
        if not data:
            return None, None
        tag = data.get("tag_name", "")
        assets = data.get("assets", [])
        for asset in assets:
            name = asset.get("name", "")
            if name.endswith(".zip"):
                url = asset.get("browser_download_url")
                if url:
                    print(f"[OK] Selected ZIP asset: {name}")
                    print(f"  Download URL: {url}")
                    return url, tag
        print("[WARNING] No zip asset found in the latest release.")
        return None, tag or None

    def check_for_updates(self, current_version):
        print("Checking for updates...")
        if not self.API_URL:
            print("Repository owner/name not configured; skip update check.")
            return None
        try:
            response = requests.get(self.API_URL, headers=self.HEADERS, timeout=15)
        except Exception as e:
            print(f"Failed to fetch release information (exception): {e}")
            return None

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
        # Prefer version.txt in app directory, but fall back to bundled updater/version.txt or module directory
        candidates = [
            self.base_dir / "version.txt",
            self.base_dir / "updater" / "version.txt",
            Path(__file__).resolve().parent / "version.txt",
        ]
        for vf in candidates:
            try:
                if vf.exists():
                    with open(vf, "r", encoding="utf-8") as f:
                        version = f.read().strip()
                        print(f"Current version from file: {version} ({vf})")
                        return version
            except Exception:
                continue
        print("No version file found (using default 0.0.0)")
        return "0.0.0"

    def download_new_version(self, download_url):
        # Create tmp_version in parent directory (same level as tmp_settings, etc.)
        parent_dir = self.parent_dir
        tmp_version_dir = parent_dir / "tmp_version"
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
                            print(f"Downloading private release with authentication...")

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

                                print(f"\nStarting download...")
                                print(f"Total size: {total / 1024 / 1024:.2f} MB")
                                start_time = time.time()

                                with open(update_zip_path, "wb") as f:
                                    for chunk in response.iter_content(chunk_size=8192):
                                        if chunk:
                                            f.write(chunk)
                                            downloaded += len(chunk)
                                            if total > 0:
                                                percent = (downloaded / total) * 100
                                                print(f"\rDownload Progress: {percent:.0f}%", end="", flush=True)
                                            else:
                                                print(f"\rDownloaded: {downloaded / 1024 / 1024:.2f} MB", end="", flush=True)

                                elapsed_time = time.time() - start_time
                                print(f"\n\n=== Download Complete ===")
                                print(f"Downloaded: {downloaded / 1024 / 1024:.2f} MB")
                                print(f"Time taken: {elapsed_time:.1f} seconds")
                                print(f"Average speed: {(downloaded / 1024 / 1024) / elapsed_time:.2f} MB/s")

                                # Verify the file exists and has content
                                print(f"\nVerifying downloaded file...")
                                if not update_zip_path.exists():
                                    print("ERROR: Downloaded file does not exist!")
                                    return None

                                file_size = update_zip_path.stat().st_size
                                if file_size == 0:
                                    print("ERROR: Downloaded file is empty!")
                                    return None

                                print(f"[OK] File verified successfully")
                                print(f"[OK] File size: {file_size / 1024 / 1024:.2f} MB")
                                print(f"[OK] File location: {update_zip_path}")
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

                print(f"\nStarting download...")
                print(f"Total size: {total / 1024 / 1024:.2f} MB")
                start_time = time.time()

                with open(update_zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                percent = (downloaded / total) * 100
                                print(f"\rDownload Progress: {percent:.0f}%", end="", flush=True)
                            else:
                                print(f"\rDownloaded: {downloaded / 1024 / 1024:.2f} MB", end="", flush=True)

                elapsed_time = time.time() - start_time
                print(f"\n\n=== Download Complete ===")
                print(f"Downloaded: {downloaded / 1024 / 1024:.2f} MB")
                print(f"Time taken: {elapsed_time:.1f} seconds")
                print(f"Average speed: {(downloaded / 1024 / 1024) / elapsed_time:.2f} MB/s")

                # Verify the file exists and has content
                print(f"\nVerifying downloaded file...")
                if not update_zip_path.exists():
                    print("ERROR: Downloaded file does not exist!")
                    return None

                file_size = update_zip_path.stat().st_size
                if file_size == 0:
                    print("ERROR: Downloaded file is empty!")
                    return None

                print(f"[OK] File verified successfully")
                print(f"[OK] File size: {file_size / 1024 / 1024:.2f} MB")
                print(f"[OK] File location: {update_zip_path}")
                return update_zip_path

        except Exception as e:
            print(f"Download exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_tmp_data(self):
        """Backup settings, videos, and data into their respective tmp folders on parent level."""
        app_dir = self.base_dir
        parent_dir = self.parent_dir
        backup_map = {
            "settings": "tmp_settings",
            "videos": "tmp_videos",
            "data": "tmp_data"
        }
        print("Backing up user data to parent directory...")
        for src, tmp in backup_map.items():
            src_path = app_dir / src
            tmp_path = parent_dir / tmp
            if src_path.exists():
                print(f"Backing up {src}...", end=" ")
                if tmp_path.exists():
                    shutil.rmtree(tmp_path)
                shutil.copytree(src_path, tmp_path)
                print(f"[OK] Backed up to {tmp}")
            else:
                print(f"Source folder {src} not found, skipping backup.")
        print("User data backup to tmp folders completed.")

    def delete_old_data(self):
        # Get the app directory from func get_project_paths
        app_dir = self.base_dir

        deleted_count = 0
        skipped_count = 0

        # Files/folders that should not be deleted
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
        extract_to = zip_path.parent  # in the same directory as zip file

        extract_to.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Starting Extraction ===")
        print(f"Source: {zip_path.name}")
        print(f"Destination: {extract_to}")

        start_time = time.time()

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.namelist()
            total_files = len(members)
            print(f"Total files to extract: {total_files}")
            print(f"Starting extraction...\n")

            for i, member in enumerate(members, 1):
                zip_ref.extract(member, extract_to)
                if i % 100 == 0 or i == total_files:
                    percent = (i / total_files) * 100
                    print(f"\rExtraction Progress: {percent:.0f}% ({i}/{total_files} files)", end="", flush=True)
            print()  # New line after progress

        elapsed_time = time.time() - start_time
        print(f"\n=== Extraction Complete ===")
        print(f"Extracted {total_files} files")
        print(f"Time taken: {elapsed_time:.1f} seconds")
        print(f"Average speed: {total_files / elapsed_time:.0f} files/second")

        # Delete zip file
        print(f"\nDeleting temporary zip file...")
        zip_path.unlink()
        print(f"[OK] Zip file deleted")

        print(f"[OK] Extraction completed successfully")

    def setup_new_version(self, tmp_version_dir):
        """Move all files from tmp_version to app directory."""
        tmp_version_dir = Path(tmp_version_dir)
        app_dir = self.base_dir

        print(f"\n=== Setting Up New Version ===")
        print(f"Source: {tmp_version_dir}")
        print(f"Destination: {app_dir}")

        # Check if there's a nested "SignAI - Desktop" folder in tmp_version
        nested_app_dir = tmp_version_dir / "SignAI - Desktop"
        if nested_app_dir.exists() and nested_app_dir.is_dir():
            print(f"[INFO] Found nested 'SignAI - Desktop' folder, using its contents")
            source_dir = nested_app_dir
        else:
            print(f"[INFO] No nested folder found, using tmp_version contents directly")
            source_dir = tmp_version_dir

        items = list(source_dir.iterdir())
        total = len(items)
        print(f"Total items to copy: {total}")
        print(f"Starting file copy operation...\n")

        start_time = time.time()
        copied_count = 0
        skipped_count = 0

        for idx, item in enumerate(items, 1):
            target = app_dir / item.name

            # Skip if it's a folder we want to preserve (will be restored later)
            if item.name in ["settings", "videos", "data"]:
                print(f"[{idx}/{total}] Skipped '{item.name}' (will be restored from backup)")
                skipped_count += 1
                continue

            try:
                print(f"[{idx}/{total}] Copying {item.name}...", end=" ")
                if item.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(item, target)
                    print(f"[OK] (directory)")
                else:
                    shutil.copy2(item, target)
                    print(f"[OK] (file)")
                copied_count += 1
            except Exception as e:
                print(f"[FAILED] {e}")

        elapsed_time = time.time() - start_time
        print(f"\n=== Setup Complete ===")
        print(f"Copied: {copied_count} items")
        print(f"Skipped: {skipped_count} items")
        print(f"Time taken: {elapsed_time:.1f} seconds")
        print(f"[OK] New version setup completed successfully")

    def delete_tmp_files(self):
        """Delete temporary folders from parent directory."""
        parent_dir = self.parent_dir

        # Add tmp_version to the list of folders to delete
        all_tmp_folders = self.tmp_folders + ["tmp_version/"]

        for folder in all_tmp_folders:
            # Remove trailing slash for consistent path handling
            folder_name = folder.rstrip('/')
            tmp_path = parent_dir / folder_name
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
                print(f"Deleted temporary folder from parent: {folder_name}")
            else:
                print(f"Temporary folder not found in parent, skipped: {folder_name}")

    def get_tmp_data(self):
        """Restore backup data from temporary folders in parent directory."""
        app_dir = self.base_dir
        parent_dir = self.parent_dir

        tmp_folders = {
            "tmp_settings": "settings",
            "tmp_videos": "videos",
            "tmp_data": "data"
        }

        print("Restoring user data from parent directory...")
        for tmp_name, target_name in tmp_folders.items():
            source = parent_dir / tmp_name
            target = app_dir / target_name

            if source.exists():
                print(f"Restoring {target_name}...", end=" ")
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
                print(f"[OK] Restored from {tmp_name}")
            else:
                print(f"Backup folder '{tmp_name}' not found in parent directory, skipping.")

    def start(self, force: bool = True):
        """Run the update process. If force=True, always install the latest release without comparing versions."""
        print("Starting updater...\n")

        if force:
            download_url, latest_tag = self.get_latest_zip_download()
            if not download_url:
                print("No download URL for latest release; aborting.")
                return
            if latest_tag:
                self.update_version_file(latest_tag)
        else:
            current_version = self.current_version()
            download_url = self.check_for_updates(current_version)
            if not download_url:
                print("No update needed.")
                return

        # 1. Backup user data to parent directory
        print("\n=== STEP 1: BACKUP USER DATA ===")
        self.create_tmp_data()

        # 2. Download new version before deleting the old one
        print("\n=== STEP 2: DOWNLOAD NEW VERSION ===")
        zip_path = self.download_new_version(download_url)
        if not zip_path:
            print("Download failed; aborting update.")
            return

        # 3. Extract new version
        print("\n=== STEP 3: EXTRACT NEW VERSION ===")
        self.unzip_new_version(zip_path)

        # 4. Delete the entire SignAI - Desktop folder
        print("\n=== STEP 4: DELETE OLD APPLICATION ===")
        app_dir = self.base_dir
        try:
            # Only proceed if we have successfully backed up user data and downloaded the new version
            if app_dir.exists():
                print(f"Deleting entire application directory: {app_dir}")
                shutil.rmtree(app_dir)
                print("Successfully deleted old application directory")

                # Recreate the empty directory
                app_dir.mkdir(parents=True)
                print("Created new empty application directory")
        except Exception as e:
            print(f"Failed to delete application directory: {e}")
            return

        # 5. Move new files to app directory
        print("\n=== STEP 5: SETUP NEW VERSION ===")
        tmp_version_dir = self.parent_dir / "tmp_version"
        self.setup_new_version(tmp_version_dir)

        # 6. Restore user data from parent directory
        print("\n=== STEP 6: RESTORE USER DATA ===")
        self.get_tmp_data()

        # 7. Verify that main exe exists before cleanup
        exe_path = app_dir / "SignAI - Desktop.exe"
        if not exe_path.exists():
            raise FileNotFoundError("Main application EXE not found after update!")

        # 8. Cleanup tmp folders AFTER everything is successful
        print("\n=== STEP 7: CLEANUP TEMPORARY FILES ===")
        self.delete_tmp_files()

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
