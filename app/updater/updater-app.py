from PySide6.QtWidgets import QApplication, QPushButton, QLabel, QTextEdit, QProgressBar
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QProcess
import re
import subprocess
import sys

# start application
app = QApplication([])

# load ui
ui_file = QFile("updater_app.ui")
ui_file.open(QFile.ReadOnly)

loader = QUiLoader()
window = loader.load(ui_file)
ui_file.close()

cancel_btn = window.findChild(QPushButton, "cancelBtn")
status_label = window.findChild(QLabel, "statusLabel")
log_text = window.findChild(QTextEdit, "logText")
progress_bar = window.findChild(QProgressBar, "progressBar")

# load qss
with open("style.qss", "r") as f:
    qss = f.read()
    app.setStyleSheet(qss)

cancel_btn.setEnabled(True)

def on_cancel_clicked():
    if status_label:
        status_label.setText("Update cancelled.")
    window.close()

cancel_btn.clicked.connect(on_cancel_clicked)

# QProcess for updater.py
process = QProcess()
process.setProgram(sys.executable)
process.setArguments(["updater.py"])
process.setWorkingDirectory(".")

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

process.start()

window.show()
app.exec()
