import os
import sys
import shutil
import subprocess
import time
import signal
import stat

APP_DIR = os.path.abspath(os.path.dirname(__file__))
UPDATER_SRC = os.path.join(APP_DIR, "updater")
TMP_UPDATER = os.path.join(APP_DIR, "tmp_updater")

def make_executable(path):
    """Macht Datei auf macOS/Linux ausführbar"""
    if os.name != "nt":  # Nur nötig auf Unix-Systemen
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)

def main():
    # copy updater in a temp folder
    try:
        shutil.copytree(UPDATER_SRC, TMP_UPDATER)
        print("Successfully copied updater.")
    except Exception as e:
        print(f"Error copying updater: {e}")
        sys.exit(1)

    # get path to updater.exe
    updater_exe = os.path.join(TMP_UPDATER, "updater.exe")
    if not os.path.isfile(updater_exe):
        print("Updater executable not found.")
        sys.exit(1)

    time.sleep(0.5)  # ensure file system is ready

    # launch updater.exe
    try:
        subprocess.Popen([updater_exe], cwd=TMP_UPDATER, close_fds=True)

    except Exception as e:
        print(f"Error executing updater: {e}")
        sys.exit(1)

    # close main exe
    try:
        # kill current process (main exe)
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        sys.exit(0)