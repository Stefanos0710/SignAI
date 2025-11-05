import os
import sys
import shutil
import subprocess
import time
import signal
import stat

# Determine app dir (when frozen use exe dir)
if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.abspath(os.path.dirname(__file__))

UPDATER_SRC = os.path.join(APP_DIR, "updater")
TMP_UPDATER = os.path.join(APP_DIR, "tmp_updater")

# Also consider onedir built updater under dist when running development
DIST_DIR = os.path.abspath(os.path.join(APP_DIR, 'dist'))
UPDATER_ONEDIR = os.path.join(DIST_DIR, 'SignAI - Updater')
UPDATER_EXE_ONEDIR = os.path.join(UPDATER_ONEDIR, 'SignAI - Updater.exe')


def make_executable(path):
    if os.name != "nt":  # only important for Unix-like systems
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)


def main():
    # Prefer running updater from onedir if present (during local runs)
    if os.path.isfile(UPDATER_EXE_ONEDIR):
        env = os.environ.copy()
        env["SIGN_AI_APP_DIR"] = APP_DIR
        try:
            subprocess.Popen([UPDATER_EXE_ONEDIR], cwd=UPDATER_ONEDIR, env=env, close_fds=True)
            os._exit(0)
        except Exception as e:
            print(f"Error executing updater from dist onedir: {e}")
            # fallback to bundled updater copy

    # Fallback: copy bundled updater folder and run its exe
    try:
        if os.path.exists(TMP_UPDATER):
            shutil.rmtree(TMP_UPDATER)
        shutil.copytree(UPDATER_SRC, TMP_UPDATER)
        print("Successfully copied updater.")
    except Exception as e:
        print(f"Error copying updater: {e}")
        sys.exit(1)

    # get path to updater.exe
    updater_exe = os.path.join(TMP_UPDATER, "SignAI - Updater.exe")
    if not os.path.isfile(updater_exe):
        print("Updater executable not found.")
        sys.exit(1)

    time.sleep(0.5)  # ensure file system is ready

    # launch updater.exe with env and cwd
    try:
        env = os.environ.copy()
        env["SIGN_AI_APP_DIR"] = APP_DIR
        subprocess.Popen([updater_exe], cwd=TMP_UPDATER, env=env, close_fds=True)
    except Exception as e:
        print(f"Error executing updater: {e}")
        sys.exit(1)

    # close main exe
    os._exit(0)
