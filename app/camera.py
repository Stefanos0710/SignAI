"""Camera module for handling camera operations.


"""

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

class Camera:
    def __init__(self, camera_id):
        self.camera_id = camera_id

    def camera_feed(self):
        pass

    def record_video(self):
        pass

    def stop_recording(self):
        pass

class CameraFeed:
    def __init__(self, label, cam_number=0):
        self.label = label
        self.cam_number = cam_number
        self.cam = cv2.VideoCapture(self.cam_number)

        # if camera could get image
        if not self.cam.isOpened():
            self.label.setText(f"No Camera Found! Cam Number: {self.cam_number}")
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20) # 20 ms = 50 fps: 1000ms/FPS = ms per second

    def update_frame(self):
        ret, frame = self.cam.read() # ret is boolean if frame is read correctly and frame is the image
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        height, width, channel = frame.shape
        bytesPerLine = channel * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg) # convert QImage to QPixmap
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))

    def start(self):
        if self.cam: # if cam active
            self.cam.release()