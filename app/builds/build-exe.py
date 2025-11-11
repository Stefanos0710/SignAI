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
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

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
    "tokenizers": (os.path.join(REPO_ROOT, "tokenizers"), "tokenizers"),
    "api": (os.path.join(REPO_ROOT, "api"), "api"),
    "models": (os.path.join(REPO_ROOT, "models"), "models"),
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
parser.add_argument("--onefile", action="store_true", help="Use --onefile (single EXE)")
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
    # default include tokenizers + api + models
    DATA_DIRS.append(CANDIDATE_DATA["tokenizers"])
    DATA_DIRS.append(CANDIDATE_DATA["api"])
    DATA_DIRS.append(CANDIDATE_DATA["models"])

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

# Default to --onedir to avoid _MEIPASS extraction; allow override with --onefile
if args.onefile:
    cmd.append("--onefile")
else:
    cmd.append("--onedir")

cmd.append(f"--name={APP_NAME}")

if ICON_PATH and os.path.exists(ICON_PATH):
    cmd.append(f"--icon={ICON_PATH}")

# add repo root for imports
if os.path.exists(REPO_ROOT):
    cmd.append(f"--paths={REPO_ROOT}")
    cmd.append(f"--paths={BASE_DIR}")

# add data dirs
added_data = []
for src, dest in DATA_DIRS:
    if os.path.exists(src):
        cmd.append(f"--add-data={src}{os.pathsep}{dest}")
        added_data.append((src, dest))
    else:
        print(f"[!] Warning: Data path not found: {src}")

# -------------------------------
# Hidden Imports and Collects
# -------------------------------
# Core application imports
hidden_imports = [
    # PySide6 core modules
    "PySide6.QtWidgets",
    "PySide6.QtUiTools",
    "PySide6.QtCore",
    "PySide6.QtGui",

    # App modules
    "camera",
    "settings",
    "videos",
    "api_call",
    "resource_path",

    # API modules
    "api.signai_api",
    "api.inference",
    "api.preprocessing_live_data",

    # TensorFlow Core
    "tensorflow",
    "tensorflow.python",
    "tensorflow.python.platform._pywrap_tf2",
    "tensorflow.python.framework.ops",
    "tensorflow.python.trackable",
    "tensorflow.python.trackable.data_structures",
    "tensorflow.python.trackable.base",
    "tensorflow.python.training.tracking",
    "tensorflow.python.eager",
    "tensorflow.python.saved_model",
    "tensorflow.python.keras",
    "tensorflow.python.keras.saving",
    "tensorflow.keras",
    "tensorflow.keras.models",
    "tensorflow.keras.layers",

    # Keras
    "keras",
    "keras.models",
    "keras.layers",
    "keras.saving",

    # Mediapipe
    "mediapipe",
    "mediapipe.python",
    "mediapipe.python.solutions",
    "mediapipe.python.solutions.holistic",

    # Matplotlib
    "matplotlib",
    "matplotlib.backends",
    "matplotlib.backends.backend_agg",
    "matplotlib._c_internal_utils",
    "matplotlib.ft2font",
    "matplotlib.pyplot",
    "matplotlib.cbook",
    "matplotlib._api",
    "matplotlib._docstring",
    "matplotlib._pylab_helpers",

    # NumPy & OpenCV
    "numpy",
    "numpy.core._methods",
    "numpy.lib.format",
    "numpy._globals",
    "numpy._distributor_init",
    "cv2",

    # Flask & Dependencies
    "flask",
    "flask_cors",
    "werkzeug",
    "werkzeug.utils",
    "jinja2",
    "click",

    # Other dependencies
    "pandas",
    "scipy",
    "requests",
    "psutil",
    "threading",
    "json",
    "pickle",
]

for hidden in hidden_imports:
    cmd.append(f"--hidden-import={hidden}")

# Collect all packages fully (OHNE mediapipe - wird manuell gehandhabt)
collects = ["tensorflow", "matplotlib", "keras", "flask"]
for pkg in collects:
    cmd.extend(["--collect-all", pkg])

# Mediapipe: Nur Python Solutions sammeln, NICHT tasks!
cmd.extend([
    "--collect-submodules=mediapipe.python",
    "--collect-data=mediapipe",  # Sammle .tflite und andere Daten
])

# Exclude problematic modules
excludes = [
    # Mediapipe tasks (alle!)
    "mediapipe.tasks",
    "mediapipe.tasks.python",
    "mediapipe.tasks.python.genai",
    "mediapipe.tasks.python.genai.converter",
    "mediapipe.tasks.python.audio",
    "mediapipe.tasks.python.audio.audio_classifier",
    "mediapipe.tasks.python.audio.core",
    "mediapipe.tasks.python.audio.core.base_audio_task_api",
    "mediapipe.tasks.python.core",
    "mediapipe.tasks.python.core.optional_dependencies",
    "mediapipe.tasks.python.vision",
    "mediapipe.tasks.python.text",
    # TensorFlow problematic
    "torch",  # not needed
    "tensorboard",  # not needed in production
    "tensorflow.tools",
    "tensorflow.tools.docs",
    "tensorflow.tools.docs.doc_controls",
    "tensorflow.python.debug",
    "tensorflow.lite.python.lite",
]
for exc in excludes:
    cmd.append(f"--exclude-module={exc}")

# -------------------------------
# Custom PyInstaller Hooks
# -------------------------------
hooks_dir = os.path.dirname(__file__)
if os.path.exists(hooks_dir):
    cmd.append(f"--additional-hooks-dir={hooks_dir}")
    print(f"+ Using custom hooks from: {hooks_dir}")

# -------------------------------
# Add DLLs for numpy / cv2 / TensorFlow
# -------------------------------
try:
    import numpy
    numpy_dir = os.path.dirname(numpy.__file__)

    # NumPy DLLs
    numpy_dll_dir = os.path.join(numpy_dir, ".libs")
    if os.path.exists(numpy_dll_dir):
        cmd.append(f"--add-binary={numpy_dll_dir}{os.pathsep}.libs")
        print(f"+ Added NumPy DLLs from {numpy_dll_dir}")

    # NumPy core libs
    numpy_core_dir = os.path.join(numpy_dir, "core")
    if os.path.exists(numpy_core_dir):
        cmd.append(f"--add-binary={numpy_core_dir}{os.pathsep}numpy/core")
        print(f"+ Added NumPy core from {numpy_core_dir}")
except ImportError:
    print("[!] NumPy could not be imported. DLLs not included.")

try:
    import cv2
    cv2_dir = os.path.dirname(cv2.__file__)

    # OpenCV DLLs
    cv2_dll_dir = os.path.join(cv2_dir, ".libs")
    if os.path.exists(cv2_dll_dir):
        cmd.append(f"--add-binary={cv2_dll_dir}{os.pathsep}.libs")
        print(f"+ Added OpenCV DLLs from {cv2_dll_dir}")
except ImportError:
    print("[!] CV2 could not be imported. DLLs not included.")

try:
    import tensorflow
    tf_dir = os.path.dirname(tensorflow.__file__)

    # TensorFlow DLLs (Windows)
    if sys.platform == "win32":
        tf_dll_dir = os.path.join(tf_dir, "python")
        if os.path.exists(tf_dll_dir):
            cmd.append(f"--add-binary={tf_dll_dir}{os.pathsep}tensorflow/python")
            print(f"+ Added TensorFlow DLLs from {tf_dll_dir}")
except ImportError:
    print("[!] TensorFlow could not be imported. DLLs not included.")

try:
    import mediapipe
    mp_dir = os.path.dirname(mediapipe.__file__)

    # Mediapipe modules
    mp_modules_dir = os.path.join(mp_dir, "modules")
    if os.path.exists(mp_modules_dir):
        cmd.append(f"--add-binary={mp_modules_dir}{os.pathsep}mediapipe/modules")
        print(f"+ Added Mediapipe modules from {mp_modules_dir}")
except ImportError:
    print("[!] Mediapipe could not be imported.")

# -------------------------------
# Runtime Hooks for better compatibility (ORDER MATTERS!)
# -------------------------------
# TensorFlow-Hook MUSS als erstes laufen!
tf_hook_path = os.path.join(os.path.dirname(__file__), "pyinstaller_hook_tensorflow.py")
if os.path.exists(tf_hook_path):
    cmd.append(f"--runtime-hook={tf_hook_path}")
    print(f"+ Using TensorFlow fix hook (priority): {tf_hook_path}")
else:
    print(f"[!] TensorFlow hook not found: {tf_hook_path}")

# Allgemeiner Runtime-Hook
runtime_hook_path = os.path.join(os.path.dirname(__file__), "pyinstaller_runtime_hook.py")
if os.path.exists(runtime_hook_path):
    cmd.append(f"--runtime-hook={runtime_hook_path}")
    print(f"+ Using runtime hook: {runtime_hook_path}")
else:
    print(f"[!] Runtime hook not found: {runtime_hook_path}")
    print(f"  App may have environment issues at startup")


# -------------------------------
# Entry File
# -------------------------------
entry_path = os.path.join(BASE_DIR, ENTRY_FILE)
if not os.path.exists(entry_path):
    print(f"[x] ERROR: Entry file not found: {entry_path}")
    sys.exit(1)

cmd.append(entry_path)

# -------------------------------
# Additional PyInstaller Options
# -------------------------------
# Disable UPX compression (can cause issues with large binaries)
cmd.append("--noupx")

# Add version info for Windows (optional)
# cmd.append("--version-file=version_info.txt")

# Optimization level
cmd.append("--optimize=2")

# -------------------------------
# Print Summary
# -------------------------------
print("=" * 80)
print(" Building SignAI Desktop EXE with PyInstaller ".center(80))
print("=" * 80)
print()
print(f"Build Mode: {'--onefile' if args.onefile else '--onedir'}")
print(f"Entry File: {entry_path}")
print(f"Output Dir: {os.path.join(BASE_DIR, 'dist')}")
print(f"Icon: {ICON_PATH if ICON_PATH and os.path.exists(ICON_PATH) else 'None'}")
print()

if added_data:
    print("Including Data:")
    for s, d in added_data:
        size = "?"
        try:
            if os.path.isfile(s):
                size = f"{os.path.getsize(s) / 1024:.1f} KB"
            elif os.path.isdir(s):
                total = sum(os.path.getsize(os.path.join(dirpath, f))
                           for dirpath, _, files in os.walk(s)
                           for f in files)
                size = f"{total / (1024*1024):.1f} MB"
        except:
            pass
        print(f"  + {s} -> {d} ({size})")
else:
    print("[!] No data directories included!")

print()
print("Command Preview:")
print(" ".join(cmd[:5]) + " ...")
print()

if args.dry_run:
    print("\nDry run mode enabled — build skipped.")
    print("\nFull Command:")
    print(" ".join(cmd))
    sys.exit(0)

# -------------------------------
# Run PyInstaller
# -------------------------------
print("=" * 80)
print(" Starting PyInstaller Build ".center(80))
print("=" * 80)
print()

pyinstaller_cmd = [sys.executable, "-m", "PyInstaller"] + cmd[1:]
try:
    result = subprocess.run(pyinstaller_cmd, cwd=BASE_DIR)
except FileNotFoundError as e:
    print("\n" + "=" * 80)
    print("[x] ERROR: Failed to start PyInstaller")
    print("=" * 80)
    print(f"\nEnsure Python and PyInstaller are installed correctly.")
    print(f"Try: pip install pyinstaller==6.16.0")
    print(f"\nError: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\n[x] Build cancelled by user")
    sys.exit(1)

# -------------------------------
# Post-Build Summary
# -------------------------------
print()
print("=" * 80)

if result.returncode == 0:
    print(" Build Completed Successfully! ".center(80, "="))
    print("=" * 80)
    print()

    if args.onefile:
        exe_path = os.path.join(BASE_DIR, "dist", f"{APP_NAME}.exe")
        print(f"[+] Executable: {exe_path}")
        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024*1024)
            print(f"  Size: {size_mb:.1f} MB")
    else:
        output_dir = os.path.join(BASE_DIR, "dist", APP_NAME)
        print(f"[+] Output folder: {output_dir}")
        exe_path = os.path.join(output_dir, f"{APP_NAME}.exe")
        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024*1024)
            print(f"  Executable: {exe_path}")
            print(f"  Exe Size: {size_mb:.1f} MB")

            # Count total size
            total_size = sum(os.path.getsize(os.path.join(dirpath, f))
                           for dirpath, _, files in os.walk(output_dir)
                           for f in files)
            print(f"  Total Size: {total_size / (1024*1024):.1f} MB")

    print()
    print("Next Steps:")
    print("  1. Test the executable")
    print("  2. Check if all features work correctly")
    print("  3. Use build-final-app.py to create distribution package")
else:
    print(" Build Failed! ".center(80, "="))
    print("=" * 80)
    print()
    print("[x] PyInstaller returned an error")
    print("  Check the logs above for details")
    print()
    print("Common issues:")
    print("  - Missing dependencies (run: pip install -r requirements.txt)")
    print("  - Incorrect paths")
    print("  - Conflicting package versions")

print("=" * 80)
