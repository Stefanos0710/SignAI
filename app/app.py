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
- **Last Update**: 2025/10/13
"""

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QPushButton, QLabel, QMainWindow, QWidget, QVBoxLayout, QCheckBox, QSpinBox, QFormLayout
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QTimer, Qt, QEvent
from networkx.algorithms.distance_measures import center

from camera import Camera, CameraFeed, findcams
from settings import Settings
from videos import HistoryVideos
from api_call import API
import sys

#setup application
app = QApplication(sys.argv)
loader = QUiLoader()

# setup settings
settings = Settings()

# setup history videos
history_videos = HistoryVideos()

# load UI
ui_file = QFile("ui/main_window.ui")
ui_file.open(QFile.ReadOnly)
window = loader.load(ui_file)
ui_file.close()

# setup icons
window.setWindowIcon(QIcon("icons/icon.png"))

# get buttons from UI
recordButton = window.findChild(QPushButton, "recordButton")
switchButton = window.findChild(QPushButton, "switchcam")
settingsButton = window.findChild(QPushButton, "settingsButton")
settingspanel = window.findChild(QWidget, "settingsPanel")
saveSettingsButton = window.findChild(QPushButton, "saveSettingsButton")
chekDebugMode = window.findChild(QCheckBox, "CheckDebugMode")
checkHistory = window.findChild(QCheckBox, "checkHistory")

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

# open style sheet
with open("style.qss", "r") as f:
    app.setStyleSheet(f.read())

# click function
def recordfunc():
    global pressed, camera, available_cams, camera_number
    pressed += 1
    print(f"PRESSED COUNT: {pressed}")

    resultDisplay.setVisible(False)

    if pressed >= 3:
        pressed = 0

    # change buttons text
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
        recordButton.setText("Start Recording")
        print("Stop Recording", pressed)

        camera.stop_recording()

        # Show result display and processing message
        resultDisplay.setVisible(True)
        resultDisplay.setPlainText("Processing video... Please wait...")
        resultDisplay.setMaximumHeight(120)

        # Call API
        api = API()
        result = api.api_translation("videos/current_video.mp4")

        # Display results in the text area
        if result.get("success"):
            translation = result.get("translation", "N/A").upper()
            confidence = result.get("confidence", 0)
            timing = result.get("timing", {})
            total_time = timing.get("total_processing_time", 0)

            # Simple 3-line display
            result_text = f"<p align='center'>Translation: {translation}</p>\n"
            result_text += f"<p align='center'>Confidence: {confidence}%</p>\n"
            result_text += f"<p align='center'>Time: {total_time}s</p>"

            resultDisplay.setHtml(result_text)
            print("Translation successful:", translation)
        else:
            error_msg = result.get("error", "Unknown error")
            resultDisplay.setPlainText(f"Translation failed\nError: {error_msg}")
            print("Translation failed:", error_msg)

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

# switch cam button function
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

# toggle settings panel
def togglesettings():
    settingspanel.setVisible(not settingspanel.isVisible())

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

# Cleanup on exit
def cleanup():
    if camerafeed:
        camerafeed.stop()
    if camera:
        camera.close()

# buttons connections/ events
recordButton.clicked.connect(recordfunc)
switchButton.clicked.connect(switchcamfunc)
settingsButton.clicked.connect(togglesettings)
checkHistory.clicked.connect(checksettings)
chekDebugMode.clicked.connect(checksettings)

app.aboutToQuit.connect(cleanup)

window.show()
sys.exit(app.exec())
