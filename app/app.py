"""
### SignAI - Sign Language translater ###

## main.py for desktop app ##

-------------------------------------------------------------
current video will be saved in app/videos/current_video.mp4
recorded videos will be saved in app/videos/history/{timestamp}_video.mp4
-------------------------------------------------------------

ToDo List:
- allways show the right camera feed when starting the app (exe)
- make debug mode show more info in the result display
- make the ai works correctly with the executable version
- make the app starts immediately

## Metadata
- **Author**: Stefanos Koufogazos Loukianov
- **Original Creation Date**: 2025/10/11
- **Last Update**: 2025/11/07
"""

# Ensure to don't import user-site packages (e.g., protobuf 6.x) when using venv
import sys as _sys
import os as _os
try:
    import site as _site
    _user_sites = _site.getusersitepackages()
    if isinstance(_user_sites, str):
        _user_sites = [_user_sites]
except Exception:
    _user_sites = []
for _p in list(_sys.path):
    try:
        if any(_os.path.abspath(_p).startswith(_os.path.abspath(up)) for up in _user_sites):
            _sys.path.remove(_p)
    except Exception:
        pass

# download and install essential packages if missing
ESSENTIAL_PACKAGES = {
    'numpy': '1.26.4',
    'protobuf': '4.25.8',
    'ml-dtypes': '0.3.1',
    'tensorflow': '2.16.2',
    'mediapipe': '0.10.21'
}

def _ensure_packages():
    import importlib, subprocess, sys
    missing = []
    for pkg, ver in ESSENTIAL_PACKAGES.items():
        try:
            importlib.import_module(pkg.replace('-', '_'))
        except Exception:
            missing.append((pkg, ver))
    if not missing:
        return
    print("[Bootstrap] Installing missing packages:", ', '.join(f"{p}=={v}" for p,v in missing))
    for pkg, ver in missing:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{pkg}=={ver}'])
        except Exception as e:
            print(f"[Bootstrap] Failed to install {pkg}: {e}")
    # Re-validate
    still_missing = []
    for pkg,_ in missing:
        try:
            importlib.import_module(pkg.replace('-', '_'))
        except Exception:
            still_missing.append(pkg)
    if still_missing:
        print("[Bootstrap] Still missing packages after attempt:", ', '.join(still_missing))
    else:
        print("[Bootstrap] All essential packages available.")

_ensure_packages()

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QPushButton, QLabel, QMainWindow, QHBoxLayout, QWidget, QVBoxLayout, QCheckBox, QSpinBox, \
    QFormLayout, QToolButton, QBoxLayout, QMessageBox
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QTimer, Qt, QEvent, QThread, Signal

from camera import Camera, CameraFeed, findcams
from settings import Settings
from videos import HistoryVideos
from api_call import API
import sys
import os
import subprocess
import platform
import time

# add resource_path import
from resource_path import resource_path, writable_path

#setup application
app = QApplication(sys.argv)
loader = QUiLoader()

# setup settings
settings = Settings()

# setup history videos
history_videos = HistoryVideos()

# load UI
ui_file_path = resource_path("ui/main_window.ui")
if not os.path.exists(ui_file_path):
    raise RuntimeError(f"UI file not found at: {ui_file_path}")

ui_file = QFile(ui_file_path)
if not ui_file.open(QFile.ReadOnly):
    raise RuntimeError(f"Unable to open UI file: {ui_file_path}")

window = loader.load(ui_file)
ui_file.close()

# setup icons
window.setWindowIcon(QIcon(resource_path("icons/icon.png")))

# get buttons from UI
recordButton = window.findChild(QPushButton, "recordButton")
switchButton = window.findChild(QPushButton, "switchcam")
settingsButton = window.findChild(QPushButton, "settingsButton")
settingspanel = window.findChild(QWidget, "settingsPanel")
saveSettingsButton = window.findChild(QPushButton, "saveSettingsButton")
chekDebugMode = window.findChild(QCheckBox, "CheckDebugMode")
checkHistory = window.findChild(QCheckBox, "checkHistory")
historybutton = window.findChild(QToolButton, "historybutton")
githubbutton = window.findChild(QToolButton, "githubButton")
updateButton = window.findChild(QPushButton, "updateButton")

# set icon to buttons
historybutton.setIcon(QIcon(resource_path("icons/history.png")))
githubbutton.setIcon(QIcon(resource_path("icons/github.png")))

# get camera feed label
videofeedlabel = window.findChild(QLabel, "videofeedlabel")

# get result display text area
resultDisplay = window.findChild(QWidget, "plainTextEdit")
resultDisplay.setReadOnly(True)
resultDisplay.setPlainText("Translation results will appear here...")
resultDisplay.setVisible(False)

# hide settings panel by default
settingspanel.setVisible(False)

# load settings and set checkboxes accordingly
checkHistory.setChecked(settings.history)
chekDebugMode.setChecked(settings.debug)

# # set header layount with History buttons, Heading and github link
# header_layout = QHBoxLayout()
# header_layout.addWidget(githubbutton)
# header_layout.addStretch()  # macht den Abstand flexibel
# header_layout.addWidget(historybutton)


# find all available cameras
available_cams = findcams()
camera_number = 0
print(f"Available Cameras: {available_cams}")

# start camera feed (use first working camera) - only for display
if available_cams:
    camerafeed = CameraFeed(videofeedlabel, cam_number=available_cams[camera_number])
else:
    videofeedlabel.setText("No working cameras found!")
    camerafeed = None

# Camera object for recording (separate from display)
camera = None

# var for clicks
pressed = 0

# globals for processing/updater
loading_timer = None
loading_dots = 0
processing_worker = None

# open style sheet
style_path = resource_path("style.qss")
if not os.path.exists(style_path):
    print(f"Warning: style file not found: {style_path}")
else:
    with open(style_path, "r") as f:
        app.setStyleSheet(f.read())

def update_loading_animation():
    global loading_dots
    try:
        loading_dots = (loading_dots + 1) % 4
        dots = '.' * loading_dots
        if resultDisplay is not None:
            resultDisplay.setPlainText('AI is thinking' + dots)
    except Exception:
        pass

def on_processing_finished(result):
    global processing_worker, loading_timer
    try:
        # stop loading timer if running
        if loading_timer is not None:
            loading_timer.stop()

        # re-enable record button
        try:
            recordButton.setEnabled(True)
            recordButton.setText('Record')
        except Exception:
            pass

        # display result
        if isinstance(result, dict):
            # try common keys
            text = result.get('text') or result.get('result') or str(result)
        else:
            text = str(result)

        if resultDisplay is not None:
            resultDisplay.setPlainText(text)
    finally:
        processing_worker = None

def on_progress_update(message: str):
    try:
        if resultDisplay is not None:
            resultDisplay.setPlainText(message)
    except Exception:
        pass


def update_loading_animation():
    global loading_dots
    try:
        loading_dots = (loading_dots + 1) % 4
        dots = '.' * loading_dots
        if resultDisplay is not None:
            resultDisplay.setPlainText('AI is thinking' + dots)
    except Exception:
        pass

def format_translation_output(result: dict, debug: bool = False) -> str:
    if not isinstance(result, dict):
        return str(result)

    translation = "Translation: " + result.get("translation") or result.get("text") or result.get("result") or "<no translation>"
    if not debug:
        return translation

    # debug output
    model = result.get("model")
    timing = result.get("timing", {})
    top = result.get("top_predictions", [])
    top_words = ", ".join(f"{p.get('word', '?')} ({p.get('confidence', 0):.2f}%)" for p in top[:5])
    confidence = result.get("confidence")
    success = result.get("success")
    video_info = result.get("video_info", {})

    return (
        f"{translation}\n\n"
        f"[Debug]\n"
        f"Succsess: {success}\n"
        f"Modell: {model}\n"
        f"Confidence: {confidence:.2f}%\n"
        f"Top 5: {top_words}\n"
        f"File: {video_info.get('filename', '?')} ({video_info.get('size_mb', '?')} MB)\n"
        f"Processing time: {timing.get('total_processing_time', '?')} s\n"
        f" - Preprocessing: {timing.get('preprocessing_time', '?')} s\n"
        f" - Inference: {timing.get('inference_time', '?')} s\n"
        f" - Model load: {timing.get('model_load_time', '?')} s"
    )


def on_processing_finished(result):
    global processing_worker, loading_timer
    try:
        if loading_timer is not None:
            loading_timer.stop()
        try:
            recordButton.setEnabled(True)
            recordButton.setText("Record")
        except Exception:
            pass

        text = format_translation_output(result, debug=settings.debug)

        if resultDisplay is not None:
            resultDisplay.setPlainText(text)
    finally:
        processing_worker = None

def on_progress_update(message: str):
    try:
        if resultDisplay is not None:
            resultDisplay.setPlainText(message)
    except Exception:
        pass

# thread for video processing and AI translation
class VideoProcessingThread(QThread):
    finished = Signal(dict)  # Signal mit Result
    progress = Signal(str)   # Signal für Status-Updates

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        self.progress.emit("Initializing API...")
        api = API()

        self.progress.emit("Uploading video...")
        result = api.api_translation(self.video_path)

        self.finished.emit(result)

def recordfunc():
    global pressed, camera, available_cams, camera_number, camerafeed, processing_worker, loading_timer
    pressed += 1
    print(f"PRESSED COUNT: {pressed}")

    resultDisplay.setVisible(False)

    if pressed >= 3:
        pressed = 0

    # check if to start or stop a recording
    if pressed == 1:
        resultDisplay.setVisible(False)
        recordButton.setText("Stop Recording")
        print("Start Recording", pressed)

        if available_cams:
            # Pass camera_feed reference to Camera so it can update display during recording
            camera = Camera(camera_id=available_cams[camera_number], camera_feed=camerafeed)
            camera.start_recording()
        else:
            print("Error: No cameras available")
            pressed = 1

    elif pressed == 2:
        # her was the start buttons text
        recordButton.setText("AI is thinking...")
        recordButton.setEnabled(False)
        print("Stop Recording", pressed)

        camera.stop_recording()

        # show and set max height for result display
        resultDisplay.setVisible(True)
        resultDisplay.setMaximumHeight(120)

        # starting the dot animation
        loading_dots = 0
        loading_timer = QTimer()
        loading_timer.timeout.connect(update_loading_animation)
        loading_timer.start(500)  # Update every 0.5 sec

        # start video processing thread
        processing_worker = VideoProcessingThread(resource_path("videos/current_video.mp4"))
        processing_worker.finished.connect(on_processing_finished)
        processing_worker.progress.connect(on_progress_update)
        processing_worker.start()

        # Save video to history if history setting is enabled
        if settings.history:
            print("Saving video to history...")
            success = history_videos.save_video()
            if success:
                print("Video successfully saved to history")
            else:
                print("Failed to save video to history")
        else:
            print("History is disabled, video not saved to history")

        pressed = 0

def switchcamfunc():
    global camera_number, camerafeed, camera

    # Stop recording if active
    if camera and camera.recording:
        camera.stop_recording()
        camera.close()
        camera = None

    # switch to next camera in list
    if camera_number < len(available_cams) - 1:
        camera_number += 1
    else:
        camera_number = 0

    # stop current camera feed
    if camerafeed:
        camerafeed.stop()


    # start new camera feed
    camerafeed = CameraFeed(videofeedlabel, cam_number=available_cams[camera_number])
    print(f"Change to camera {camera_number}: {available_cams[camera_number]}")

def togglesettings():
    settingspanel.setVisible(not settingspanel.isVisible())

    if settingspanel.isVisible():
        settingsButton.setText("Close Settings")
    else:
        settingsButton.setText("Settings")

def checksettings():
    # check history settings
    if checkHistory.isChecked():
        settings.history = True
        print("History Enabled")
    elif not checkHistory.isChecked():
        settings.history = False
        print("History Disabled")

    # check debug mode settings
    if chekDebugMode.isChecked():
        settings.debug = True
        print("Debug Mode Enabled")
    elif not chekDebugMode.isChecked():
        settings.debug = False
        print("Debug Mode Disabled")

    # Save settings to file
    settings.save()

def cleanup():
    if camerafeed:
        camerafeed.stop()
    if camera:
        camera.close()

def historyfunc():
    system = platform.system()

    path = os.path.abspath(writable_path("videos/history"))

    try:
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"Failed to open history: {e}")

def githubfunc():
    import webbrowser
    webbrowser.open("https://github.com/Stefanos0710/SignAI/")

def start_update():
    """Start the updater application and close the main app."""
    print("[Update] Button clicked")

    # Determine app dir (frozen uses exe dir)
    if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.abspath('.')

    print(f"[Update] app_dir={app_dir}")

    parent = os.path.dirname(app_dir)
    grandparent = os.path.dirname(parent)

    # Kandidatenpfade für Updater-EXE
    candidates = [
        # gleiche Ebene wie die App
        os.path.join(app_dir, "SignAI - Updater.exe"),
        os.path.join(app_dir, "SignAI - Updater", "SignAI - Updater.exe"),
        os.path.join(app_dir, "updater", "SignAI - Updater.exe"),
        # Geschwister unter parent (typisch: dist/SignAI - Desktop und dist/SignAI - Updater)
        os.path.join(parent, "SignAI - Updater.exe"),
        os.path.join(parent, "SignAI - Updater", "SignAI - Updater.exe"),
        # eine Ebene höher (z. B. app/final vs. app/dist)
        os.path.join(grandparent, "SignAI - Updater.exe"),
        os.path.join(grandparent, "SignAI - Updater", "SignAI - Updater.exe"),
    ]

    updater_exe_path = next((p for p in candidates if os.path.isfile(p)), None)

    if updater_exe_path:
        try:
            env = os.environ.copy()
            env["SIGN_AI_APP_DIR"] = app_dir
            print(f"[Update] Using updater: {updater_exe_path}")
            print(f"[Update] SIGN_AI_APP_DIR = {app_dir}")
            cwd = os.path.dirname(updater_exe_path)
            subprocess.Popen([updater_exe_path], env=env, cwd=cwd)
            print("[Update] Updater started successfully.")
            QTimer.singleShot(100, lambda: sys.exit(0))
            app.quit()
            return
        except Exception as e:
            print(f"[Update] Error starting updater: {e}")
            import traceback; traceback.print_exc()
            try:
                QMessageBox.critical(None, "Update", f"Updater konnte nicht gestartet werden:\n{e}")
            except Exception:
                pass
            # continue to dev fallback

    # Dev-Fallback: starte start_updater.py (kopiert updater/ und startet EXE dort)
    start_updater_py = resource_path("start_updater.py")
    if os.path.exists(start_updater_py):
        print(f"[Update] Fallback to start_updater.py: {start_updater_py}")
        try:
            env = os.environ.copy()
            env["SIGN_AI_APP_DIR"] = app_dir
            subprocess.Popen([sys.executable, start_updater_py], env=env, cwd=os.path.dirname(start_updater_py))
            QTimer.singleShot(100, lambda: sys.exit(0))
            app.quit()
            return
        except Exception as e:
            print(f"[Update] start_updater.py failed: {e}")

    # Nichts gefunden -> Meldung
    msg = (
        "Updater wurde nicht gefunden. Erwartet unter:\n" +
        "\n".join(candidates) +
        "\n\nOder Fallback 'start_updater.py' ist nicht ausführbar."
    )
    print(msg)
    try:
        QMessageBox.warning(None, "Update", msg)
    except Exception:
        pass

# buttons connections/ events
recordButton.clicked.connect(recordfunc)
switchButton.clicked.connect(switchcamfunc)
settingsButton.clicked.connect(togglesettings)
checkHistory.clicked.connect(checksettings)
chekDebugMode.clicked.connect(checksettings)
historybutton.clicked.connect(historyfunc)
githubbutton.clicked.connect(githubfunc)
updateButton.clicked.connect(start_update)

app.aboutToQuit.connect(cleanup)

window.show()
sys.exit(app.exec())
