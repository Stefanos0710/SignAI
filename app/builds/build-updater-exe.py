import os
import sys
import subprocess
import argparse

# ---------------------------
# Configuration
# ---------------------------

APP_NAME = "SignAI - Updater"
ENTRY_FILE = os.path.join("updater", "updater-app.py")

# esample:C:\Users\stefa\Documents\GitHub\SignAI\app
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("-"*40)
print(BASE_DIR)
print("-"*40)

# Icon (try .ico, fallback to .png)
ICON_PATH = os.path.join(BASE_DIR, "icons", "icon.ico")
if not os.path.exists(ICON_PATH):
    alt_png = os.path.join(BASE_DIR, "icons", "icon.png")
    ICON_PATH = alt_png if os.path.exists(alt_png) else None

# Data that should always be bundled
DATA_DIRS = [
    (os.path.join(BASE_DIR, "updater"), "updater"),
    (os.path.join(BASE_DIR, "updater", "updater_app.ui"), "."),
    (os.path.join(BASE_DIR, "ui"), "ui"),
    (os.path.join(BASE_DIR, "icons"), "icons"),
    (os.path.join(BASE_DIR, "updater", "style.qss"), "."),
    (os.path.join(BASE_DIR, "settings"), "settings"),
    (os.path.join(BASE_DIR, "updater", "version.txt"), "."),
]

parser = argparse.ArgumentParser(description='Build SignAI Updater EXE using PyInstaller')
parser.add_argument('--onefile', action='store_true', help='Build as one-file executable (default is onedir)')
parser.add_argument('--dry-run', action='store_true', help='Print the PyInstaller command and exit')
parser.add_argument('--clean', action='store_true', help='Remove previous build/dist folders')
args = parser.parse_args()

cmd = ["pyinstaller", "--noconsole"]
cmd.append(f"--name={APP_NAME}")
# Default to --onedir to avoid PyInstaller temp extraction
cmd.append("--onefile" if args.onefile else "--onedir")

if ICON_PATH:
    cmd.append(f"--icon={ICON_PATH}")

# Add repo path for external modules
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
cmd.append(f"--paths={REPO_ROOT}")

# Add data
for src, dest in DATA_DIRS:
    if os.path.exists(src):
        cmd.append(f"--add-data={src}{os.pathsep}{dest}")

# hidden imports (PySide6, requests, dotenv)
hidden_imports = [
    "PySide6.QtWidgets",
    "PySide6.QtUiTools",
    "PySide6.QtCore",
    "requests",
    "dotenv",
]

for hidden in hidden_imports:
    cmd.append(f"--hidden-import={hidden}")

# Matplotlib ft2font (only keep specific hidden import if required)
cmd.append("--hidden-import=matplotlib.ft2font")

# Entry File
entry_path = os.path.join(BASE_DIR, ENTRY_FILE)
cmd.append(entry_path)

print("Building Updater EXE with PyInstaller...")
print("Command:", " ".join(cmd))

if args.dry_run:
    sys.exit(0)

result = subprocess.run([sys.executable, "-m", "PyInstaller"] + cmd[1:], cwd=BASE_DIR)
if result.returncode == 0:
    print(f"Build success! EXE folder is in: {os.path.join(BASE_DIR, 'dist', APP_NAME)}")
else:
    print("Build failed, check errors above.")
