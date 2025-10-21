import os
import sys
import subprocess
import argparse

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

# default candidate data dirs (absolute src path, dest inside package)
CANDIDATE_DATA = {
    'ui': (os.path.join(BASE_DIR, "ui"), "ui"),
    'icons': (os.path.join(BASE_DIR, "icons"), "icons"),
    'videos': (os.path.join(BASE_DIR, "videos"), "videos"),
    'style': (os.path.join(BASE_DIR, "style.qss"), "."),
    'settings': (os.path.join(BASE_DIR, "settings"), "settings"),
    'version': (os.path.join(BASE_DIR, "version.txt"), "."),
    'tokenizers': (os.path.abspath(os.path.join(BASE_DIR, "..", "tokenizers")), "tokenizers"),
    'api': (os.path.abspath(os.path.join(BASE_DIR, "..", "api")), "api"),
    'models': (os.path.abspath(os.path.join(BASE_DIR, "..", "models")), "models"),
}

# -------------------------------------
# CLI Arguments
# -------------------------------------
parser = argparse.ArgumentParser(description='Build SignAI desktop EXE using PyInstaller')
parser.add_argument('--include-models', action='store_true', help='Include ../models in the build (may be large)')
parser.add_argument('--include-tokenizers', action='store_true', help='Include ../tokenizers')
parser.add_argument('--include-api', action='store_true', help='Include ../api folder')
parser.add_argument('--include-webside', action='store_true', help='Include webside folders (product_webside, webside_application)')
parser.add_argument('--onedir', action='store_true', help='Use --onedir (useful for debugging) instead of --onefile')
parser.add_argument('--dry-run', action='store_true', help='Print the pyinstaller command and exit without running it')
parser.add_argument('--clean', action='store_true', help='Remove previous build/dist folders before building')
args = parser.parse_args()

# determine which data entries to include
DATA_DIRS = []
# always include these if present
for key in ('ui', 'icons', 'videos', 'style', 'settings', 'version'):
    src, dest = CANDIDATE_DATA[key]
    DATA_DIRS.append((src, dest))

# conditional includes
if args.include_tokenizers or args.include_api or args.include_models or args.include_webside:
    if args.include_tokenizers:
        DATA_DIRS.append(CANDIDATE_DATA['tokenizers'])
    if args.include_api:
        DATA_DIRS.append(CANDIDATE_DATA['api'])
    if args.include_models:
        DATA_DIRS.append(CANDIDATE_DATA['models'])
    if args.include_webside:
        DATA_DIRS.append(CANDIDATE_DATA['product_webside'])
        DATA_DIRS.append(CANDIDATE_DATA['webside_application'])
else:
    # defaults (include api and tokenizers by default for parity with previous script)
    DATA_DIRS.append(CANDIDATE_DATA['tokenizers'])
    DATA_DIRS.append(CANDIDATE_DATA['api'])

# -------------------------------------
# BUILD COMMAND
# -------------------------------------
cmd = [
    "pyinstaller",
    "--noconsole",
]
if args.onedir:
    cmd.append("--onedir")
else:
    cmd.append("--onefile")

cmd.append(f"--name={APP_NAME}")

# add icon if exists
if ICON_PATH and os.path.exists(ICON_PATH):
    cmd.append(f"--icon={ICON_PATH}")

# add repo root to PyInstaller search paths so modules/packages outside app/ (like api/) are discovered
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
if os.path.exists(REPO_ROOT):
    cmd.append(f"--paths={REPO_ROOT}")

# add data entries (only those present on disk)
added_data = []
for src, dest in DATA_DIRS:
    if os.path.exists(src):
        cmd.append(f"--add-data={src}{os.pathsep}{dest}")
        added_data.append((src, dest))

# entry point
entry_path = os.path.join(BASE_DIR, ENTRY_FILE)
cmd.append(entry_path)

# clean previous builds if requested
if args.clean:
    import shutil
    for folder in ('build', 'dist'):
        path = os.path.join(BASE_DIR, folder)
        if os.path.exists(path):
            print(f"Removing {path}...")
            shutil.rmtree(path)

# show summary
print("Building EXE with PyInstaller...\n")
print("Command:")
print(" ".join(cmd))

if added_data:
    print("\nIncluding the following data entries:")
    for s, d in added_data:
        print(f" - {s} -> {d}")
else:
    print("\nNo additional data directories/files found to include.")

if args.dry_run:
    print('\nDry run requested, not invoking PyInstaller.')
    sys.exit(0)

hidden_imports = [
    "numpy.core._methods",
    "numpy.lib.format",
    "numpy._globals",
    "numpy._distributor_init",
    "cv2",
    "numpy",
]

for hidden in hidden_imports:
    cmd.append(f"--hidden-import={hidden}")

# --- Numpy DLLs einbinden ---
try:
    import numpy
    numpy_dir = os.path.dirname(numpy.__file__)
    numpy_dll_dir = os.path.join(numpy_dir, '.libs')
    if os.path.exists(numpy_dll_dir):
        cmd.append(f'--add-binary={numpy_dll_dir}{os.pathsep}.libs')
except ImportError:
    print("Warnung: numpy konnte nicht importiert werden. DLLs werden nicht explizit eingebunden.")

# --- OpenCV DLLs einbinden (optional, falls benötigt) ---
try:
    import cv2
    cv2_dir = os.path.dirname(cv2.__file__)
    cv2_dll_dir = os.path.join(cv2_dir, '.libs')
    if os.path.exists(cv2_dll_dir):
        cmd.append(f'--add-binary={cv2_dll_dir}{os.pathsep}.libs')
except ImportError:
    print("Warnung: cv2 konnte nicht importiert werden. DLLs werden nicht explizit eingebunden.")

# Run PyInstaller
result = subprocess.run(cmd, cwd=BASE_DIR)

if result.returncode == 0:
    print("\nBuild successfully!")
    if args.onedir:
        print(f"Your build folder is: {os.path.join(BASE_DIR, 'dist', APP_NAME)}")
    else:
        print(f"Your EXE is here: dist\\{APP_NAME}.exe")
else:
    print("\nBuild failed. Please check the error messages above.")
