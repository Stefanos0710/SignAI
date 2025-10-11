"""
### SignAI - Sign Language translater ###

## PC-APP ##

-------------------------------------------------------------

here comes the description of the app

-------------------------------------------------------------

## Metadata
- **Author**: Stefanos Koufogazos Loukianov
- **Original Creation Date**: 2025/10/11
- **Last Update**: 2025/10/11
"""
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QPushButton, QLabel, QMainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QTimer
from camera import Camera, CameraFeed
import sys

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

# get camera feed label
videofeedlabel = window.findChild(QLabel, "videofeedlabel")

# start camera feed
camerafeed = CameraFeed(videofeedlabel, cam_number=0)

# var for clicks
pressed = 0

# click function
def recordfunc():
    global pressed # use gloabal var
    pressed += 1

    # change buttons text
    if pressed == 0:
        recordButton.setText("Start Recording")
        print("Start Recording", pressed)

    elif pressed == 1:
        recordButton.setText("Stop Recording")
        print("Stop Recording", pressed)

    elif pressed == 2:
        recordButton.setText("AI is Thinking...")
        print("AI is Thinking...", pressed)
        pressed = -1

        # Here comes the AI API call

recordButton.clicked.connect(recordfunc)

window.show()
app.exec()

camerafeed.start()