"""
### SignAI - Sign Language translater ###

## main.py for desktop app ##

-------------------------------------------------------------
current video will be saved in app/videos/current_video.mp4
recorded videos will be saved in app/videos/history/{timestamp}_video.mp4
-------------------------------------------------------------

Todos:
- [ ] create settings
- [ ] Integrate AI API for sign language translation

## Metadata
- **Author**: Stefanos Koufogazos Loukianov
- **Original Creation Date**: 2025/10/11
- **Last Update**: 2025/10/11
"""

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QPushButton, QLabel, QMainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QTimer
from camera import Camera, CameraFeed, findcams
import sys
import time

app = QApplication(sys.argv)
loader = QUiLoader()

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

# get camera feed label
videofeedlabel = window.findChild(QLabel, "videofeedlabel")

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

with open("style.qss", "r") as f:
    app.setStyleSheet(f.read())

# click function
def recordfunc():
    global pressed, camera, available_cams, camera_number
    pressed += 1
    print(f"PRESSED COUNT: {pressed} --------------------------------------------------------------------")

    # change buttons text
    if pressed == 1:
        recordButton.setText("Stop Recording")
        print("Start Recording", pressed)

        if available_cams:
            camera = Camera(camera_id=available_cams[camera_number])
            camera.start_recording()
        else:
            print("Error: No cameras available")
            pressed = 1


    elif pressed == 2:
        recordButton.setText("Start Recording")
        print("Stop Recording", pressed)

        camera.stop_recording()
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

recordButton.clicked.connect(recordfunc)
switchButton.clicked.connect(switchcamfunc)

# Cleanup on exit
def cleanup():
    if camerafeed:
        camerafeed.stop()
    if camera:
        camera.close()

app.aboutToQuit.connect(cleanup)

window.show()
sys.exit(app.exec())
