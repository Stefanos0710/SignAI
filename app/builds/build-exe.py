import os
import sys
import subprocess
import argparse
import shutil

"""
=============================================================================
 SignAI - Desktop Build Script
 Build your EXE using PyInstaller
 Compatible with Python 3.10–3.12
=============================================================================
"""

# -------------------------------
# Configuration
# -------------------------------
APP_NAME = "SignAI - Desktop"
ENTRY_FILE = "app.py"

# compute base dir (app folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# icon path (prefers .ico, fallback .png)
ICON_PATH = os.path.join(BASE_DIR, "icons", "icon.ico")
if not os.path.exists(ICON_PATH):
    alt_png = os.path.join(BASE_DIR, "icons", "icon.png")
    ICON_PATH = alt_png if os.path.exists(alt_png) else None

# candidate data directories and files
CANDIDATE_DATA = {
    "ui": (os.path.join(BASE_DIR, "ui"), "ui"),
    "icons": (os.path.join(BASE_DIR, "icons"), "icons"),
    "videos": (os.path.join(BASE_DIR, "videos"), "videos"),
    "style": (os.path.join(BASE_DIR, "style.qss"), "."),
    "settings": (os.path.join(BASE_DIR, "settings"), "settings"),
    "version": (os.path.join(BASE_DIR, "version.txt"), "."),
    "start_updater": (os.path.join(BASE_DIR, "start_updater.py"), "."),
    "updater": (os.path.join(BASE_DIR, "updater"), "updater"),
    "tokenizers": (os.path.abspath(os.path.join(BASE_DIR, "..", "tokenizers")), "tokenizers"),
    "api": (os.path.abspath(os.path.join(BASE_DIR, "..", "api")), "api"),
    "models": (os.path.abspath(os.path.join(BASE_DIR, "..", "models")), "models"),
}

# -------------------------------
# CLI Arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Build SignAI Desktop EXE using PyInstaller")
parser.add_argument("--include-models", action="store_true", help="Include ../models in build (large)")
parser.add_argument("--include-tokenizers", action="store_true", help="Include ../tokenizers")
parser.add_argument("--include-api", action="store_true", help="Include ../api folder")
parser.add_argument("--include-webside", action="store_true", help="Include webside folders")
parser.add_argument("--onedir", action="store_true", help="Use --onedir (debug mode)")
parser.add_argument("--dry-run", action="store_true", help="Show command but do not run")
parser.add_argument("--clean", action="store_true", help="Clean old build/dist folders")
args = parser.parse_args()

# -------------------------------
# Determine Data Directories
# -------------------------------
DATA_DIRS = []
# always include these
for key in ("ui", "icons", "videos", "style", "settings", "version", "updater"):
    DATA_DIRS.append(CANDIDATE_DATA[key])

# Conditional includes
if args.include_tokenizers or args.include_api or args.include_models or args.include_webside:
    if args.include_tokenizers:
        DATA_DIRS.append(CANDIDATE_DATA["tokenizers"])
    if args.include_api:
        DATA_DIRS.append(CANDIDATE_DATA["api"])
    if args.include_models:
        DATA_DIRS.append(CANDIDATE_DATA["models"])
else:
    # default include tokenizers + api
    DATA_DIRS.append(CANDIDATE_DATA["tokenizers"])
    DATA_DIRS.append(CANDIDATE_DATA["api"])

# -------------------------------
# Clean old builds
# -------------------------------
if args.clean:
    for folder in ("build", "dist"):
        path = os.path.join(BASE_DIR, folder)
        if os.path.exists(path):
            print(f"Removing {path}...")
            shutil.rmtree(path)

# -------------------------------
# PyInstaller Command
# -------------------------------
cmd = ["pyinstaller", "--noconsole"]

cmd.append("--onedir" if args.onedir else "--onefile")
cmd.append(f"--name={APP_NAME}")

if ICON_PATH and os.path.exists(ICON_PATH):
    cmd.append(f"--icon={ICON_PATH}")

# add repo root for imports
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if os.path.exists(REPO_ROOT):
    cmd.append(f"--paths={REPO_ROOT}")

# add data dirs
added_data = []
for src, dest in DATA_DIRS:
    if os.path.exists(src):
        cmd.append(f"--add-data={src}{os.pathsep}{dest}")
        added_data.append((src, dest))

# -------------------------------
# Hidden Imports and Collects
# -------------------------------
# TensorFlow, Mediapipe, Matplotlib, NumPy, OpenCV
hidden_imports = [
    # TensorFlow Core
    "tensorflow.python.platform._pywrap_tf2",
    "tensorflow.python",
    "tensorflow.python.framework.ops",
    "tensorflow.python.trackable",
    "tensorflow.python.trackable.data_structures",
    "tensorflow.python.trackable.base",
    "tensorflow.python.training.tracking",
    # Mediapipe
    "mediapipe",
    # Matplotlib
    "matplotlib._c_internal_utils",
    "matplotlib.ft2font",
    "matplotlib.backends",
    "matplotlib.pyplot",
    "matplotlib.cbook",
    "matplotlib._api",
    "matplotlib._docstring",
    "matplotlib._pylab_helpers",
    # NumPy / OpenCV
    "numpy.core._methods",
    "numpy.lib.format",
    "numpy._globals",
    "numpy._distributor_init",
    "cv2",
    "numpy",
]

for hidden in hidden_imports:
    cmd.append(f"--hidden-import={hidden}")

# Collect all packages fully
collects = ["tensorflow", "mediapipe", "matplotlib"]
for pkg in collects:
    cmd.extend(["--collect-all", pkg, "--collect-submodules", pkg])

# Exclude Mediapipe AI converter (benötigt torch)
cmd.append("--exclude-module=mediapipe.tasks.python.genai.converter")

# -------------------------------
# Add DLLs for numpy / cv2
# -------------------------------
try:
    import numpy

    numpy_dir = os.path.dirname(numpy.__file__)
    numpy_dll_dir = os.path.join(numpy_dir, ".libs")
    if os.path.exists(numpy_dll_dir):
        cmd.append(f"--add-binary={numpy_dll_dir}{os.pathsep}.libs")
except ImportError:
    print("Numpy could not be imported. DLLs not included.")

try:
    import cv2

    cv2_dir = os.path.dirname(cv2.__file__)
    cv2_dll_dir = os.path.join(cv2_dir, ".libs")
    if os.path.exists(cv2_dll_dir):
        cmd.append(f"--add-binary={cv2_dll_dir}{os.pathsep}.libs")
except ImportError:
    print("CV2 could not be imported. DLLs not included.")

# -------------------------------
# Entry File
# -------------------------------
entry_path = os.path.join(BASE_DIR, ENTRY_FILE)
cmd.append(entry_path)

# -------------------------------
# Print Summary
# -------------------------------
print("==============================================")
print(" Building SignAI Desktop EXE with PyInstaller ")
print("==============================================\n")
print("Command:")
print(" ".join(cmd))

if added_data:
    print("Including Data:")
    for s, d in added_data:
        print(f"  - {s} -> {d}")
else:
    print("No data directories included.")

if args.dry_run:
    print("\nDry run mode enabled — build skipped.")
    sys.exit(0)

# -------------------------------
# Run PyInstaller
# -------------------------------
result = subprocess.run(cmd, cwd=BASE_DIR)

if result.returncode == 0:
    print("\nBuild successful!")
    if args.onedir:
        print(f"Output folder: {os.path.join(BASE_DIR, 'dist', APP_NAME)}")
    else:
        print(f"Executable: dist\\{APP_NAME}.exe")
else:
    print("\nBuild failed! Check the logs above.")
