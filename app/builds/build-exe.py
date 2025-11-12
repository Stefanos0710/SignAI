import os
import sys
import shutil
import stat
import argparse
import subprocess
import time

"""
=============================================================================
 SignAI - Desktop Build Script
 Build your EXE using PyInstaller
 Compatible with Python 3.10–3.12
=============================================================================

Execution (Windows CMD):
    cd app\builds
    python build-exe.py             # normal build
    python build-exe.py --dry-run   # just show
    python build-exe.py --clean     # delete build/ and dist/
    python build-exe.py --onefile   # build in one file (not recommended)

Output:
    app/dist/SignAI - Desktop/SignAI - Desktop.exe
or with --onefile:
    app/dist/SignAI - Desktop.exe

"""

APP_NAME = "SignAI - Desktop"
ENTRY_FILE = "app.py"  # is at the app/app.py level
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # .../app
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # project root for api package
DIST_DIR = os.path.join(BASE_DIR, "dist")
BUILD_DIR = os.path.join(BASE_DIR, "build")
TARGET_ONEDIR = os.path.join(DIST_DIR, APP_NAME)
TARGET_ONEFILE = os.path.join(DIST_DIR, f"{APP_NAME}.exe")

ICON_CANDIDATES = [
    os.path.join(BASE_DIR, "icons", "icon.ico"),
    os.path.join(BASE_DIR, "icons", "icon.png")
]
ICON_PATH = next((p for p in ICON_CANDIDATES if os.path.exists(p)), None)

# Needed Data files and folders
DATA_ITEMS = [
    (os.path.join(BASE_DIR, "ui", "main_window.ui"), "ui"),
    (os.path.join(BASE_DIR, "icons"), "icons"),
    (os.path.join(BASE_DIR, "style.qss"), "."),
    (os.path.join(BASE_DIR, "settings", "settings.json"), "settings"),
    # Tokenizer & Modelle
    (os.path.join(REPO_ROOT, "tokenizers", "gloss_tokenizer.json"), "tokenizers"),
]
# get models from models folder
MODELS_DIR = os.path.join(REPO_ROOT, "models")
if os.path.isdir(MODELS_DIR):
    for f in os.listdir(MODELS_DIR):
        if f.lower().endswith(".keras") or f.lower().endswith(".h5"):
            DATA_ITEMS.append((os.path.join(MODELS_DIR, f), "models"))
    # Trainings-History Bilder optional
    for f in os.listdir(MODELS_DIR):
        if f.lower().endswith(".png"):
            DATA_ITEMS.append((os.path.join(MODELS_DIR, f), "models"))

# be sure that the settings folder exists and can be later been written to
os.makedirs(os.path.join(BASE_DIR, "settings"), exist_ok=True)

# get extra folders which are needed at runtime
EXTRA_DIRS = [
    os.path.join(BASE_DIR, "videos"),
    os.path.join(BASE_DIR, "videos", "history"),
    os.path.join(REPO_ROOT, "data", "live"),
    os.path.join(REPO_ROOT, "data", "live", "video"),
]
for d in EXTRA_DIRS:
    os.makedirs(d, exist_ok=True)

# placeholder files
_placeholders = [
    os.path.join(BASE_DIR, "videos", "current_video.mp4"),  # empty video placeholder
    os.path.join(BASE_DIR, "videos", "history", ".keep"),
    os.path.join(REPO_ROOT, "data", "live", "live_dataset.csv"),
    os.path.join(REPO_ROOT, "data", "live", "video", ".keep"),
]
for ph in _placeholders:
    if not os.path.exists(ph):
        try:
            with open(ph, "wb") as _f:
                _f.write(b"")
        except Exception:
            pass

# add extra data items to be sure
DATA_ITEMS.extend([
    (os.path.join(BASE_DIR, "videos"), "videos"),
    (os.path.join(BASE_DIR, "videos", "history"), os.path.join("videos", "history")),
    (os.path.join(REPO_ROOT, "data", "live"), os.path.join("data", "live")),
    (os.path.join(REPO_ROOT, "data", "live", "video"), os.path.join("data", "live", "video")),
])

# hidden imports
HIDDEN_IMPORTS = [
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtUiTools",
    "PySide6.QtXml",
    "PySide6.QtNetwork",
    "PySide6.QtSvg",
    "cv2",
    "mediapipe",
    "numpy",
    "requests",
    "psutil",
    "flask",
    "werkzeug",
    "jinja2",
    "threading",  # to be sure threading is included
    # Liberies for inferenz & Data
    "tensorflow",
    "keras",
    "ml_dtypes",
    "pandas",
    "scipy",
    "json",
    "csv",
    "resource_path",  # add for more stable resource handling
    # get lokal liberies/models
    "camera",
    "videos",
    "settings",
    "api_call",
    # api packag modules
    "api.signai_api",
    "api.preprocessing_live_data",
    "api.inference",
    # additionally flask modules
    "flask_cors",
    "flask_socketio",
]

# optional hidden imports because of failing portential matplotlib usage
OPTIONAL_HIDDEN = [
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.ft2font",
]

# delete unneeded packages to reduce size
EXCLUDES = [
    # "seaborn", "pandas"
]

# collects all data/plugins from these packages
COLLECT_ALL_PKGS = [
    "PySide6", "mediapipe", "cv2", "flask", "werkzeug", "jinja2", "flask_cors", "flask_socketio",
    "tensorflow", "keras"
]

parser = argparse.ArgumentParser(description="PyInstaller build for SignAI Desktop")
parser.add_argument("--dry-run", action="store_true", help="Just show the build command without executing it")
parser.add_argument("--clean", action="store_true", help="delete dist/ and build/ directories before")
parser.add_argument("--onefile", action="store_true", help="Build one single EXE file (not recommended)")
args = parser.parse_args()


def _rmtree(path: str):
    if not os.path.exists(path):
        return
    def onerror(func, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass
    shutil.rmtree(path, onerror=onerror)


def clean():
    print("[CLEAN] Deleting build/ and dist/ directories...")
    _rmtree(BUILD_DIR)
    _rmtree(DIST_DIR)


def remove_target_conflict():
    # removing existing target to avoid PyInstaller errors
    if os.path.isdir(TARGET_ONEDIR):
        print(f"[CLEAN] Removing target onedir: {TARGET_ONEDIR}")
        _rmtree(TARGET_ONEDIR)
    if os.path.isfile(TARGET_ONEFILE):
        print(f"[CLEAN] Removing old exe: {TARGET_ONEFILE}")
        try:
            os.remove(TARGET_ONEFILE)
        except Exception:
            pass


def build_command():
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        f"--name={APP_NAME}",
        "--windowed",
    ]

    # onedir against onefile mode
    if args.onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    if ICON_PATH:
        cmd.append(f"--icon={ICON_PATH}")

    # path to search for imports
    cmd.append(f"--paths={REPO_ROOT}")

    # add data files
    for src, dest in DATA_ITEMS:
        if os.path.exists(src):
            cmd.append(f"--add-data={src}{os.pathsep}{dest}")
        else:
            print(f"[WARN] Datendatei fehlt: {src}")

    # hidden imports
    for hidden in HIDDEN_IMPORTS + OPTIONAL_HIDDEN:
        cmd.append(f"--hidden-import={hidden}")

    # collect all packages, modules, data
    for pkg in COLLECT_ALL_PKGS:
        cmd.append(f"--collect-all={pkg}")
    # optional specific submodules
    cmd.append("--collect-submodules=PySide6")

    # Excludes
    for exc in EXCLUDES:
        cmd.append(f"--exclude-module={exc}")

    # entry point
    entry_path = os.path.join(BASE_DIR, ENTRY_FILE)
    cmd.append(entry_path)
    return cmd


def main():
    print("=" * 60)
    print("SignAI Desktop Build")
    print("=" * 60)
    print(f"Basic Dir: {BASE_DIR}")
    print(f"Repo base path:      {REPO_ROOT}")
    print(f"Icon:             {ICON_PATH or '(no Icon found)'}")
    print(f"Mode:            {'ONEFILE' if args.onefile else 'ONEDIR'}")

    if args.clean:
        clean()
    else:
        remove_target_conflict()

    cmd = build_command()
    print("\n[CMD] Calling Pyinstaller:")
    print(" ".join(cmd))

    if args.dry_run:
        print("\n[DRY-RUN] No build running.")
        return

    start = time.time()
    result = subprocess.run([sys.executable, "-m", "PyInstaller"] + cmd[1:], cwd=BASE_DIR)
    duration = time.time() - start

    if result.returncode != 0:
        print(f"\n[FEHLER] Build failed! Time: {duration:.1f}s")
        sys.exit(result.returncode)

    print(f"\n[FERTIG] Build successful: Time: {duration:.1f}s")
    if args.onefile:
        if os.path.isfile(TARGET_ONEFILE):
            print(f"Output file: {TARGET_ONEFILE}")
        else:
            print("[WARN] Did not found the built EXE, please check.")
    else:
        if os.path.isdir(TARGET_ONEDIR):
            print(f"Output Dir: {TARGET_ONEDIR}")
            print("Hint: Zip the whole folder for distribution.")
        else:
            print("[WARN] Destination folder not found, please check.")

    print("\nNext Steps:")
    print("  1. Test the Exe in the folder dist/")
    print("  2. If you ran into antivirus issues, consider adding an exception for the folder.")
    print("  3. For final version run build-final-app.py to create also the updater.")


if __name__ == "__main__":
    main()
