from PySide6.QtWidgets import QApplication, QPushButton, QLabel, QTextEdit, QProgressBar
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QProcess
import re
import subprocess
import sys
import os
import shutil

# start application
app = QApplication([])

# get paths
base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
ui_file_path = os.path.join(base_path, "updater_app.ui")
ui_file = QFile(ui_file_path)
ui_file.open(QFile.ReadOnly)

loader = QUiLoader()
window = loader.load(ui_file)
ui_file.close()

cancel_btn = window.findChild(QPushButton, "cancelBtn")
status_label = window.findChild(QLabel, "statusLabel")
log_text = window.findChild(QTextEdit, "logText")
progress_bar = window.findChild(QProgressBar, "progressBar")

# load qss
style_path = os.path.join(base_path, "style.qss")
with open(style_path, "r") as f:
    qss = f.read()
    app.setStyleSheet(qss)

cancel_btn.setEnabled(True)

def on_cancel_clicked():
    if status_label:
        status_label.setText("Update cancelled.")
    window.close()

cancel_btn.clicked.connect(on_cancel_clicked)

# QProcess for updater.py
if getattr(sys, 'frozen', False):
    python_exe = os.path.join(os.path.dirname(sys.executable), "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"  # Fallback
else:
    python_exe = sys.executable

# get path to updater.py
updater_py_path = os.path.join(base_path, "updater.py")
if not os.path.exists(updater_py_path):
    updater_py_path = os.path.join(base_path, "updater", "updater.py")

# setup updater.py in frozen state
if not os.path.exists(updater_py_path):
    src_updater = os.path.join(base_path, "updater", "updater.py")
    dst_updater = os.path.join(os.path.dirname(sys.executable), "updater.py")
    if os.path.exists(src_updater):
        shutil.copy(src_updater, dst_updater)
        updater_py_path = dst_updater

process = QProcess()
process.setProgram(python_exe)
process.setArguments([updater_py_path])
process.setWorkingDirectory(base_path)

# start once
if process.state() != QProcess.Running:
    process.start()

def handle_stdout():
    data = process.readAllStandardOutput().data().decode("utf-8")
    if log_text:
        log_text.append(data)
    # extract progress percentage from output
    match = re.search(r"Progress: (\d+)%", data)
    if match and progress_bar:
        percent = int(match.group(1))
        progress_bar.setValue(percent)

def handle_stderr():
    data = process.readAllStandardError().data().decode("utf-8")
    if log_text:
        log_text.append(data)

process.readyReadStandardOutput.connect(handle_stdout)
process.readyReadStandardError.connect(handle_stderr)

try:
    import dotenv
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])

window.show()
app.exec()
