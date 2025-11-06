import os
import sys
import shutil
import subprocess
import time
import argparse

# CLI args to allow dry-run/clean orchestration
parser = argparse.ArgumentParser(description='Orchestrate final builds for SignAI')
parser.add_argument('--dry-run', action='store_true', help='Run inner build scripts in dry-run mode')
parser.add_argument('--clean', action='store_true', help='Pass --clean to inner build scripts')
args = parser.parse_args()

# folder for final builds
FINAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'final'))
if not os.path.exists(FINAL_DIR):
    os.makedirs(FINAL_DIR)

# delete previous build dirs
def handle_remove_readonly(func, path, exc):
    import stat
    excvalue = exc[1]
    if func in (os.unlink, os.remove) and excvalue.errno == 5:
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception as e:
            print(f"Konnte nicht löschen: {path} ({e}) - Datei wird übersprungen.")
    else:
        print(f"Fehler beim Löschen: {path} ({excvalue}) - Datei wird übersprungen.")

def clean_build_dirs():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for folder in ['dist', 'build']:
        path = os.path.join(base_dir, folder)
        if os.path.exists(path):
            print(f"Delete dictory: {path}")
            shutil.rmtree(path, onerror=handle_remove_readonly)

# func to copy build output
def copy_build_output(src_path, dest_name):
    if os.path.exists(src_path):
        dest_path = os.path.join(FINAL_DIR, dest_name)
        if os.path.isdir(src_path):
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)
        print(f"Copy: {src_path} -> {dest_path}")
    else:
        print(f"Missing: {src_path}")

def main():
    if os.name == "nt":
        subprocess.run(["taskkill", "/f", "/im", "SignAI - Desktop.exe"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        time.sleep(2)  # wat a moment to ensure process is closed

    # delete previous build dirs before build
    if args.clean:
        clean_build_dirs()

    # build desktop app
    print("Build desktop-app...")
    # build-exe: pass --dry-run/--clean flags if requested
    build_exe_cmd = [sys.executable, 'build-exe.py']
    if args.dry_run:
        build_exe_cmd.append('--dry-run')
    if args.clean:
        build_exe_cmd.append('--clean')
    result = subprocess.run(build_exe_cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print("error building desktop app!")
        sys.exit(1)

    # copy output
    dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dist'))
    # prefer onedir folder copy; if onefile was built, exe will exist instead
    onedir_folder = os.path.join(dist_dir, 'SignAI - Desktop')
    if os.path.isdir(onedir_folder):
        copy_build_output(onedir_folder, 'SignAI - Desktop')
    else:
        exe_path = os.path.join(dist_dir, 'SignAI - Desktop.exe')
        copy_build_output(exe_path, 'SignAI - Desktop.exe')

    # # delete previous build dirs before next build
    # clean_build_dirs()

    # build updater exe
    print("Build updater...")
    build_updater_cmd = [sys.executable, 'build-updater-exe.py']
    if args.dry_run:
        build_updater_cmd.append('--dry-run')
    if args.clean:
        build_updater_cmd.append('--clean')
    result = subprocess.run(build_updater_cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print("error building updater!")
        sys.exit(1)

    # Updater now defaults to onedir folder
    updater_folder = os.path.join(dist_dir, 'SignAI - Updater')
    if not args.dry_run:
        if os.path.isdir(updater_folder):
            copy_build_output(updater_folder, 'SignAI - Updater')
        else:
            updater_exe_path = os.path.join(dist_dir, 'SignAI - Updater.exe')
            copy_build_output(updater_exe_path, 'SignAI - Updater.exe')
    else:
        print(f"Dry-run: would copy {updater_folder} (or exe fallback) to final folder")
    print("Finished! The final exe´s are in /builds/final/")

if __name__ == "__main__":
    main()
