from PySide6.QtWidgets import QApplication, QPushButton, QLabel, QTextEdit, QProgressBar
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QProcess
import re
import subprocess
import sys
import os
import shutil

# Headless mode: run the updater logic directly and exit
if "--run-updater" in sys.argv:
    try:
        # Delay import to avoid PySide6 GUI setup in headless mode
        from updater.updater import Updater
        up = Updater()
        up.start()
        sys.exit(0)
    except Exception as e:
        print(f"[Updater-Headless] Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# start application
app = QApplication([])

# get paths: prefer executable directory in frozen mode (onedir); otherwise module dir
if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

print(f"[Updater-UI] base_path: {base_path}")

# Helper: recursive find of a filename under base_path
def find_file_recursive(root, filename, max_depth=3):
    try:
        for cur_root, dirs, files in os.walk(root):
            # depth check
            depth = os.path.relpath(cur_root, root).count(os.sep)
            if depth > max_depth:
                # don't go too deep
                dirs[:] = []
                continue
            if filename in files:
                return os.path.join(cur_root, filename)
    except Exception as e:
        print(f"[Updater-UI] find_file_recursive error: {e}")
    return None

# Resolve UI path
candidate_ui_paths = [
    os.path.join(base_path, "updater_app.ui"),
    os.path.join(base_path, "updater", "updater_app.ui"),
]
ui_file_path = next((p for p in candidate_ui_paths if os.path.exists(p)), None)
if not ui_file_path:
    ui_file_path = find_file_recursive(base_path, "updater_app.ui")

print(f"[Updater-UI] ui_file_path: {ui_file_path}")

if not ui_file_path or not os.path.exists(ui_file_path):
    # Log directory listing for diagnostics
    try:
        print(f"[Updater-UI] Listing base_path: {base_path}")
        print(os.listdir(base_path))
        updater_dir = os.path.join(base_path, "updater")
        if os.path.isdir(updater_dir):
            print(f"[Updater-UI] Listing updater dir: {updater_dir}")
            print(os.listdir(updater_dir))
    except Exception as e:
        print(f"[Updater-UI] Could not list directories: {e}")
    raise RuntimeError(
        "Updater UI-Datei (updater_app.ui) wurde nicht gefunden. Bitte stelle sicher, dass sie im Ordner vorhanden ist."
    )

ui_file = QFile(ui_file_path)
if not ui_file.open(QFile.ReadOnly):
    # Additional diagnostics
    try:
        print(f"[Updater-UI] QFile.open failed for: {ui_file_path}")
        print(f"[Updater-UI] os.path.exists -> {os.path.exists(ui_file_path)}")
    except Exception:
        pass
    raise RuntimeError(f"Unable to open UI file: {ui_file_path}")

loader = QUiLoader()
window = loader.load(ui_file)
ui_file.close()

if window is None:
    raise RuntimeError(f"Unable to load UI from: {ui_file_path}")

cancel_btn = window.findChild(QPushButton, "cancelBtn")
status_label = window.findChild(QLabel, "statusLabel")
log_text = window.findChild(QTextEdit, "logText")
progress_bar = window.findChild(QProgressBar, "progressBar")

# load qss
style_candidates = [
    os.path.join(base_path, "style.qss"),
    os.path.join(base_path, "updater", "style.qss"),
]
if not any(os.path.exists(p) for p in style_candidates):
    # try recursive search for style.qss
    alt_style = find_file_recursive(base_path, "style.qss")
    if alt_style:
        style_candidates.insert(0, alt_style)

style_path = next((p for p in style_candidates if os.path.exists(p)), style_candidates[0])
if os.path.exists(style_path):
    with open(style_path, "r") as f:
        qss = f.read()
        app.setStyleSheet(qss)
else:
    print(f"[Updater-UI] Warning: style file not found: {style_path}")

cancel_btn.setEnabled(True)

def on_cancel_clicked():
    if status_label:
        status_label.setText("Update cancelled.")
    window.close()

cancel_btn.clicked.connect(on_cancel_clicked)

# QProcess to run the same EXE in headless mode
process = QProcess()
if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
    program = sys.executable
    args = ["--run-updater"]
else:
    program = sys.executable
    args = [os.path.abspath(__file__), "--run-updater"]

process.setProgram(program)
process.setArguments(args)
process.setWorkingDirectory(base_path)

# start once
if process.state() != QProcess.Running:
    process.start()

# Wire stdout/stderr to the UI

def handle_stdout():
    data = process.readAllStandardOutput().data().decode("utf-8", errors="replace")
    if log_text and data:
        log_text.append(data)
    # try parse percent if present (dev mode), else ignore
    m = re.search(r"(\d+)%", data)
    if m and progress_bar:
        try:
            progress_bar.setValue(int(m.group(1)))
        except Exception:
            pass

def handle_stderr():
    data = process.readAllStandardError().data().decode("utf-8", errors="replace")
    if log_text and data:
        log_text.append(data)

process.readyReadStandardOutput.connect(handle_stdout)
process.readyReadStandardError.connect(handle_stderr)

try:
    import dotenv
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])

window.show()
app.exec()
