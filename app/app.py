"""
### SignAI - Sign Language translater ###

## main.py for desktop app ##

-------------------------------------------------------------
current video will be saved in app/videos/current_video.mp4
recorded videos will be saved in app/videos/history/{timestamp}_video.mp4
-------------------------------------------------------------

## Metadata
- **Author**: Stefanos Koufogazos Loukianov
- **Original Creation Date**: 2025/10/11
- **Last Update**: 2025/11/11
"""

import sys, os, io, traceback, types, socket, subprocess, platform, time

APP_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) and hasattr(sys, 'executable') else os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
print('[Bootstrap] APP_DIR:', APP_DIR)

# Clean user-site packages ONLY in non-frozen mode (avoid picking wrong user packages like protobuf 6.x)
if not (getattr(sys, 'frozen', False) and hasattr(sys, 'executable')):
    try:
        import site as _site
        _user_sites = _site.getusersitepackages()
        if isinstance(_user_sites, str):
            _user_sites = [_user_sites]
    except Exception:
        _user_sites = []
    for _p in list(sys.path):
        try:
            # remove paths that are within user site-packages
            if any(up and os.path.abspath(_p).startswith(os.path.abspath(up) + os.sep)
                   or os.path.abspath(_p) == os.path.abspath(up)
                   for up in _user_sites):
                sys.path.remove(_p)
        except Exception:
            pass

# Essential packages (install only in dev, not in frozen)
ESSENTIAL_PACKAGES = {
    'numpy': '1.26.4',
    'protobuf': '4.25.8',
    'ml-dtypes': '0.3.1',
    'tensorflow': '2.16.2',
    'mediapipe': '0.10.21'
}
def _ensure_packages():
    import importlib, subprocess
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

if not getattr(sys, 'frozen', False):
    _ensure_packages()

# resource_path + writable_path (with stub if missing)
try:
    from resource_path import resource_path, writable_path
    print('[Bootstrap] resource_path module loaded.')
except Exception as _e_rp_import:
    print('[Bootstrap] resource_path missing, create stub:', _e_rp_import)
    def resource_path(relative_path: str) -> str:
        base = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) and hasattr(sys, 'executable') else os.path.dirname(os.path.abspath(__file__))
        full = os.path.join(base, relative_path)
        if os.path.exists(full):
            return full
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            alt = os.path.join(meipass, relative_path)
            if os.path.exists(alt):
                return alt
        return full
    def writable_path(relative_path: str) -> str:
        base = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) and hasattr(sys, 'executable') else os.path.dirname(os.path.abspath(__file__))
        full = os.path.join(base, relative_path)
        parent = os.path.dirname(full)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return full
    stub_mod = types.ModuleType('resource_path')
    stub_mod.resource_path = resource_path
    stub_mod.writable_path = writable_path
    sys.modules['resource_path'] = stub_mod

# Ensure videos/ directories exist (write location)
os.makedirs(os.path.dirname(writable_path('videos/current_video.mp4')), exist_ok=True)
os.makedirs(writable_path('videos/history'), exist_ok=True)

# Logging: Tee to file + console (so logs appear in dev and are saved in exe)
class TeeOutput:
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

def setup_logging():
    try:
        log_dir = writable_path('logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, 'desktop_app.log')
        log_file = open(log_file_path, 'a', encoding='utf-8', buffering=1)
        # Tee: to original console (if available) + file
        sys.stdout = TeeOutput(getattr(sys, '__stdout__', None), log_file)
        sys.stderr = TeeOutput(getattr(sys, '__stderr__', None), log_file)
        print(f"[Bootstrap] Logging enabled (tee) -> {log_file_path}")
    except Exception as e:
        # If anything goes wrong, keep default stdout/stderr
        print("[Bootstrap] Failed to set up logging:", e)

setup_logging()

# Single-Instance lock
_SINGLE_INSTANCE_PORT = 52391
try:
    _s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _s.bind(('127.0.0.1', _SINGLE_INSTANCE_PORT))
    _s.listen(1)
    print(f"[Singleton] Lock on port {_SINGLE_INSTANCE_PORT}")
except Exception:
    print("[Singleton] Another instance is already running. Exit.")
    sys.exit(0)

print(f"[Bootstrap] Frozen: {getattr(sys, 'frozen', False)}; Ensure-packages active: {not getattr(sys, 'frozen', False)}")

try:
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QPushButton, QLabel, QWidget, QToolButton, QMessageBox, QCheckBox, QPlainTextEdit, QTextEdit
    from PySide6.QtUiTools import QUiLoader
    from PySide6.QtCore import QFile, QTimer, Qt, QThread, Signal, QIODevice
except Exception as _e_qt_init:
    print('[Fatal] Qt imports failed:', _e_qt_init)
    traceback.print_exc()
    sys.exit(1)

def import_with_fallback(module_name, rel_path=None):
    try:
        mod = __import__(module_name)
        print(f'[Bootstrap] {module_name} loaded.')
        return mod
    except ModuleNotFoundError as e:
        import importlib.util
        print(f'[Error] {module_name} not found:', e)
        path = os.path.join(APP_DIR, rel_path if rel_path else f"{module_name}.py")
        if os.path.isfile(path):
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules[module_name] = mod
                print(f'[Bootstrap] {module_name} loaded via file path.')
                return mod
            except Exception as e2:
                print(f'[Fatal] Loading {module_name}.py failed:', e2)
        return None

# camera
camera_mod = import_with_fallback('camera', 'camera.py')
if camera_mod:
    Camera = getattr(camera_mod, 'Camera', None)
    CameraFeed = getattr(camera_mod, 'CameraFeed', None)
    findcams = getattr(camera_mod, 'findcams', lambda: [])
else:
    class Camera:
        def __init__(self, *a, **k): pass
        def start_recording(self): print('[StubCamera] start_recording')
        def stop_recording(self): print('[StubCamera] stop_recording')
        def close(self): pass
        recording = False
    class CameraFeed:
        def __init__(self, *a, **k): pass
        def stop(self): pass
    def findcams(): return []

settings_mod = import_with_fallback('settings', 'settings.py')
if settings_mod and hasattr(settings_mod, 'Settings'):
    Settings = settings_mod.Settings
else:
    class Settings:
        def __init__(self):
            self.folder = 'settings'
            self.debug = False
            self.history = True
        def save(self):
            try:
                os.makedirs(self.folder, exist_ok=True)
                with open(os.path.join(self.folder, 'settings.json'), 'w', encoding='utf-8') as f:
                    f.write('{"debug": false, "history": true}')
            except Exception: pass

videos_mod = import_with_fallback('videos', 'videos.py')
if videos_mod and hasattr(videos_mod, 'HistoryVideos'):
    HistoryVideos = videos_mod.HistoryVideos
else:
    class HistoryVideos:
        def save_video(self):
            return False

try:
    from api_call import API
except Exception as e:
    print('[Error] api_call import failed:', e)
    class API:
        def api_translation(self, video_path):
            return {"success": False, "error": "API unavailable", "translation": ""}

# =========================
# App init
# =========================
app = QApplication(sys.argv)
loader = QUiLoader()

settings = Settings()
history_videos = HistoryVideos()

# Load UI
ui_file_path = resource_path("ui/main_window.ui")
if not os.path.exists(ui_file_path):
    print(f"[Fatal] UI file not found: {ui_file_path}")
    sys.exit(1)
ui_file = QFile(ui_file_path)
if not ui_file.open(QIODevice.ReadOnly):
    print(f"[Fatal] Cannot open UI file: {ui_file_path}")
    sys.exit(1)
window = loader.load(ui_file)
ui_file.close()
if window is None:
    print('[Fatal] UI loader returned None.')
    sys.exit(1)

# Icon
try:
    window.setWindowIcon(QIcon(resource_path("icons/icon.png")))
except Exception as _e_icon:
    print('[Warn] Could not set icon:', _e_icon)

# UI elements
recordButton   = window.findChild(QPushButton,    "recordButton")
switchButton   = window.findChild(QPushButton,    "switchcam")
settingsButton = window.findChild(QPushButton,    "settingsButton")
settingspanel  = window.findChild(QWidget,        "settingsPanel")
chekDebugMode  = window.findChild(QCheckBox,      "CheckDebugMode")
checkHistory   = window.findChild(QCheckBox,      "checkHistory")
historybutton  = window.findChild(QToolButton,    "historybutton")
githubbutton   = window.findChild(QToolButton,    "githubButton")
updateButton   = window.findChild(QPushButton,    "updateButton")
videofeedlabel = window.findChild(QLabel,         "videofeedlabel")
# Try QPlainTextEdit first, then QTextEdit (UI uses QTextEdit)
resultDisplay  = window.findChild(QPlainTextEdit, "plainTextEdit") or window.findChild(QTextEdit, "plainTextEdit")

# Initial state of result display: hidden and read-only
if resultDisplay:
    try:
        if hasattr(resultDisplay, 'setReadOnly'):
            resultDisplay.setReadOnly(True)
        if hasattr(resultDisplay, 'clear'):
            resultDisplay.clear()
        resultDisplay.setVisible(False)
    except Exception as _e_init_display:
        print('[Warn] Could not init result display:', _e_init_display)

# Settings panel hidden by default
if settingspanel:
    settingspanel.setVisible(False)

# Load settings -> apply to checkboxes
if checkHistory:  checkHistory.setChecked(getattr(settings, 'history', True))
if chekDebugMode: chekDebugMode.setChecked(getattr(settings, 'debug', False))

# Button icons
try:
    if historybutton: historybutton.setIcon(QIcon(resource_path("icons/history.png")))
    if githubbutton:  githubbutton.setIcon(QIcon(resource_path("icons/github.png")))
except Exception:
    pass

# Style
style_path = resource_path("style.qss")
if os.path.exists(style_path):
    try:
        with open(style_path, 'r', encoding='utf-8') as f:
            app.setStyleSheet(f.read())
    except Exception as e:
        print('[Warn] Could not load style.qss:', e)

# Cameras
available_cams = findcams()
print(f"[Camera] Available: {available_cams}")
camera_number = 0
camerafeed = CameraFeed(videofeedlabel, cam_number=available_cams[camera_number]) if available_cams else None
if not available_cams and videofeedlabel:
    videofeedlabel.setText("No working cameras found!")

camera = None
pressed = 0
loading_timer = None
loading_dots = 0
processing_worker = None

# =========================
# API worker
# =========================
from PySide6.QtCore import QObject

class VideoProcessingThread(QThread):
    finished = Signal(dict)  # result dict
    progress = Signal(str)   # progress messages

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            self.progress.emit("Initializing API...")
            api = API()
            self.progress.emit("Uploading video...")
            result = api.api_translation(self.video_path)
            if not isinstance(result, dict):
                result = {"success": True, "translation": str(result)}
            self.finished.emit(result)
        except Exception as e:
            err = ''.join(traceback.format_exception_only(type(e), e)).strip()
            self.finished.emit({"success": False, "error": err, "translation": ""})

def update_loading_animation():
    global loading_dots
    try:
        loading_dots = (loading_dots + 1) % 4
        dots = '.' * loading_dots
        info_text = (
            f"AI is thinking{dots}\n"
            "Please wait a moment, prediction may take a few seconds."
        )
        if resultDisplay is not None and resultDisplay.isVisible():
            if hasattr(resultDisplay, 'setPlainText'):
                resultDisplay.setPlainText(info_text)
            else:
                resultDisplay.setText(info_text)
    except Exception:
        pass

def format_translation_output(result: dict, debug: bool = False) -> str:
    if not isinstance(result, dict):
        return str(result)

    # show error + any translation provided
    if not result.get('success', True) and result.get('error'):
        base = result.get('translation') or result.get('text') or result.get('result') or '<no translation>'
        return f"Error: {result.get('error')}\nTranslation: {base}"

    base = result.get('translation') or result.get('text') or result.get('result') or '<no translation>'
    translation = f"Translation: {base}"

    if not debug:
        return translation

    # Debug details
    timing = result.get('timing', {})
    top = result.get('top_predictions', [])
    top_words = ", ".join(f"{p.get('word', '?')} ({p.get('confidence', 0):.2f}%)" for p in top[:5])
    video_info = result.get('video_info', {})
    return (
        f"{translation}\n\n[Debug]\n"
        f"Success: {result.get('success')}\n"
        f"Model: {result.get('model')}\n"
        f"Confidence: {result.get('confidence', 0):.2f}%\n"
        f"Top 5: {top_words}\n"
        f"File: {video_info.get('filename', '?')} ({video_info.get('size_mb', '?')} MB)\n"
        f"Processing time: {timing.get('total_processing_time', '?')} s\n"
        f" - Preprocessing: {timing.get('preprocessing_time', '?')} s\n"
        f" - Inference: {timing.get('inference_time', '?')} s\n"
        f" - Model load: {timing.get('model_load_time', '?')} s"
    )

def on_processing_finished(result: dict):
    global processing_worker, loading_timer
    try:
        if loading_timer:
            loading_timer.stop()
            loading_timer = None

        if recordButton:
            try:
                recordButton.setEnabled(True)
                recordButton.setText('Record')
            except Exception:
                pass

        text = format_translation_output(result, debug=getattr(settings, 'debug', False))
        if resultDisplay is not None:
            try:
                resultDisplay.setVisible(True)  # ensure visible when showing result
                if hasattr(resultDisplay, 'setPlainText'):
                    resultDisplay.setPlainText(text)
                else:
                    resultDisplay.setText(text)
            except Exception as _e_set_text:
                print('[Warn] Could not update result display:', _e_set_text)
    finally:
        processing_worker = None

def on_progress_update(message: str):
    try:
        if resultDisplay is not None:
            if not resultDisplay.isVisible():
                resultDisplay.setVisible(True)
                resultDisplay.setMaximumHeight(140)
            if hasattr(resultDisplay, 'setPlainText'):
                resultDisplay.setPlainText(message)
            else:
                resultDisplay.setText(message)
    except Exception:
        pass

def recordfunc():
    global pressed, camera, available_cams, camera_number, camerafeed, processing_worker, loading_timer, loading_dots
    pressed += 1
    print(f"[Record] Pressed: {pressed}")

    if pressed >= 3:
        pressed = 0

    # Start recording
    if pressed == 1:
        # Hide translation panel at start (fix "always open")
        if resultDisplay:
            try:
                # ensure it's empty and hidden and non-editable
                if hasattr(resultDisplay, 'clear'):
                    resultDisplay.clear()
                resultDisplay.setVisible(False)
                if hasattr(resultDisplay, 'setReadOnly'):
                    resultDisplay.setReadOnly(True)
            except Exception:
                pass

        if recordButton:
            recordButton.setText('Stop Recording')

        if available_cams:
            camera = Camera(camera_id=available_cams[camera_number], camera_feed=camerafeed)
            camera.start_recording()
        else:
            print('[Record] No cameras available.')
            pressed = 1

    # Stop and process
    elif pressed == 2:
        if recordButton:
            recordButton.setText('AI is thinking...')
            recordButton.setEnabled(False)

        if camera:
            camera.stop_recording()

        # Verify video file exists and has size
        video_path = writable_path('videos/current_video.mp4')
        if not os.path.exists(video_path):
            print(f"[Error] Video not created: {video_path}")
            if recordButton:
                recordButton.setText("Error: No Video")
                recordButton.setEnabled(True)
            pressed = 0
            return

        size = os.path.getsize(video_path)
        if size == 0:
            print(f"[Error] Video empty: {video_path}")
            if recordButton:
                recordButton.setText("Error: Empty Video")
                recordButton.setEnabled(True)
            pressed = 0
            return

        print(f"[Record] Video OK: {size} bytes")

        # Show result panel and start loading animation
        if resultDisplay:
            resultDisplay.setVisible(True)
            resultDisplay.setMaximumHeight(140)
            # ensure non-editable while showing progress
            try:
                if hasattr(resultDisplay, 'setReadOnly'):
                    resultDisplay.setReadOnly(True)
            except Exception:
                pass
        loading_dots = 0
        loading_timer = QTimer()
        loading_timer.timeout.connect(update_loading_animation)
        loading_timer.start(500)

        # Start API worker (v2-style)
        processing_worker = VideoProcessingThread(video_path)
        processing_worker.finished.connect(on_processing_finished)
        processing_worker.progress.connect(on_progress_update)
        processing_worker.start()

        # Save to history (if enabled)
        if getattr(settings, 'history', True):
            ok = history_videos.save_video()
            print('[History] Saved =', ok)
        else:
            print('[History] Disabled')

        pressed = 0

def switchcamfunc():
    global camera_number, camerafeed, camera
    if camera and getattr(camera, 'recording', False):
        camera.stop_recording()
        camera.close()
        camera = None

    if available_cams:
        camera_number = (camera_number + 1) % len(available_cams)
        if camerafeed:
            camerafeed.stop()
        camerafeed = CameraFeed(videofeedlabel, cam_number=available_cams[camera_number])
        print(f"[Camera] Switched to {camera_number}: {available_cams[camera_number]}")

def togglesettings():
    if not settingspanel:
        return
    settingspanel.setVisible(not settingspanel.isVisible())
    if settingsButton:
        settingsButton.setText('Close Settings' if settingspanel.isVisible() else 'Settings')

def checksettings():
    if checkHistory:  settings.history = checkHistory.isChecked()
    if chekDebugMode: settings.debug   = chekDebugMode.isChecked()
    settings.save()
    print(f"[Settings] history={settings.history} debug={settings.debug}")

def cleanup():
    try:
        if camerafeed: camerafeed.stop()
        if camera: camera.close()
    except Exception:
        pass

def historyfunc():
    path = os.path.abspath(writable_path('videos/history'))
    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(path)
        elif system == 'Darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
    except Exception as e:
        print('[History] Open failed:', e)

def githubfunc():
    import webbrowser
    try:
        webbrowser.open('https://github.com/Stefanos0710/SignAI/')
    except Exception as e:
        print('[GitHub] Open failed:', e)

def start_update():
    """Start updater executable if found, else try start_updater.py"""
    print('[Update] Triggered')
    app_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) and hasattr(sys, 'executable') else os.path.abspath('.')
    parent = os.path.dirname(app_dir)
    grandparent = os.path.dirname(parent)
    candidates = [
        os.path.join(app_dir, 'SignAI - Updater.exe'),
        os.path.join(app_dir, 'SignAI - Updater', 'SignAI - Updater.exe'),
        os.path.join(app_dir, 'updater', 'SignAI - Updater.exe'),
        os.path.join(parent, 'SignAI - Updater.exe'),
        os.path.join(parent, 'SignAI - Updater', 'SignAI - Updater.exe'),
        os.path.join(grandparent, 'SignAI - Updater.exe'),
        os.path.join(grandparent, 'SignAI - Updater', 'SignAI - Updater.exe'),
    ]
    updater_exe_path = next((p for p in candidates if os.path.isfile(p)), None)
    if updater_exe_path:
        try:
            env = os.environ.copy()
            env['SIGN_AI_APP_DIR'] = app_dir
            subprocess.Popen([updater_exe_path], env=env, cwd=os.path.dirname(updater_exe_path))
            QTimer.singleShot(150, lambda: sys.exit(0))
            return
        except Exception as e:
            print('[Update] Starting updater EXE failed:', e)
    fallback = resource_path('start_updater.py')
    if os.path.exists(fallback):
        try:
            env = os.environ.copy()
            env['SIGN_AI_APP_DIR'] = app_dir
            subprocess.Popen([sys.executable, fallback], env=env, cwd=os.path.dirname(fallback))
            QTimer.singleShot(150, lambda: sys.exit(0))
            return
        except Exception as e:
            print('[Update] Fallback start_updater.py failed:', e)
    print('[Update] No updater found.')

# =========================
# Connect UI
# =========================
if recordButton:   recordButton.clicked.connect(recordfunc)
if switchButton:   switchButton.clicked.connect(switchcamfunc)
if settingsButton: settingsButton.clicked.connect(togglesettings)
if checkHistory:   checkHistory.clicked.connect(checksettings)
if chekDebugMode:  chekDebugMode.clicked.connect(checksettings)
if historybutton:  historybutton.clicked.connect(historyfunc)
if githubbutton:   githubbutton.clicked.connect(githubfunc)
if updateButton:   updateButton.clicked.connect(start_update)

app.aboutToQuit.connect(cleanup)

# =========================
# Run Application
# =========================
window.show()
try:
    exit_code = app.exec()
    print(f"[Shutdown] Qt Event loop exited with code {exit_code}")
    sys.exit(exit_code)
except Exception as _e_loop:
    print('[Fatal] Crash in event loop:', _e_loop)
    traceback.print_exc()
    sys.exit(1)
