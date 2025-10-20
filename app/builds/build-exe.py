import os
import sys
import subprocess

""" **Configuration** """

# Main Application Settings
APP_NAME = "SignAI - Desktop"
ENTRY_FILE = "app.py"

# compute base dir (app folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# icon path (try .ico first, fallback to .png)
ICON_PATH = os.path.join(BASE_DIR, "icons", "icon.ico")
if not os.path.exists(ICON_PATH):
    alt_png = os.path.join(BASE_DIR, "icons", "icon.png")
    if os.path.exists(alt_png):
        ICON_PATH = alt_png
    else:
        ICON_PATH = None

# Data directories to include (source absolute path, destination inside the package)
DATA_DIRS = [
    (os.path.join(BASE_DIR, "ui"), "ui"),
    (os.path.join(BASE_DIR, "icons"), "icons"),
    (os.path.join(BASE_DIR, "videos"), "videos"),
    (os.path.join(BASE_DIR, "style.qss"), "."),
    # include app settings folder if present
    (os.path.join(BASE_DIR, "settings"), "settings"),
    # include version file
    (os.path.join(BASE_DIR, "version.txt"), "."),
    # include tokenizers from repo root (use parent of app)
    (os.path.abspath(os.path.join(BASE_DIR, "..", "tokenizers")), "tokenizers"),
    # include api backend (repo root)
    (os.path.abspath(os.path.join(BASE_DIR, "..", "api")), "api"),
    # include models folder if present (may be large)
    (os.path.abspath(os.path.join(BASE_DIR, "..", "models")), "models"),
]

# -------------------------------------
# BUILD COMMAND
# -------------------------------------
cmd = [
    "pyinstaller",
    "--noconsole",
    "--onefile",
    f"--name={APP_NAME}",
]

# add icon if exists
if ICON_PATH and os.path.exists(ICON_PATH):
    cmd.append(f"--icon={ICON_PATH}")

# add data
added_data = []
for src, dest in DATA_DIRS:
    if os.path.exists(src):
        # Use os.pathsep so the script works on Windows (;) and *nix (:)
        cmd.append(f"--add-data={src}{os.pathsep}{dest}")
        added_data.append((src, dest))

# entry point (use absolute path inside app folder)
entry_path = os.path.join(BASE_DIR, ENTRY_FILE)
cmd.append(entry_path)

# -------------------------------------
# Start Build
# -------------------------------------
print("Building EXE with PyInstaller...\n")
print("Command:")
print(" ".join(cmd))

if added_data:
    print("\nIncluding the following data entries:")
    for s, d in added_data:
        print(f" - {s} -> {d}")
else:
    print("\nNo additional data directories/files found to include.")

# Run PyInstaller in the app base directory so resources are resolved consistently
result = subprocess.run(cmd, cwd=BASE_DIR)

if result.returncode == 0:
    print("\nBuild successfully!")
    print(f"Your EXE is here: dist\\{APP_NAME}.exe")
else:
    print("\nBuild failed. Please check the error messages above.")
