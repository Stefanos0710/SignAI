"""Camera module for handling camera operations.


"""

import cv2
import os
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

# Suppress OpenCV warnings about camera indices
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

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
        self.cam = cv2.VideoCapture(self.cam_number, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        self.timer = None
        self.is_running = False

        # Check if camera could get image
        if not self.cam.isOpened():
            self.label.setText(f"No Camera Found! Cam Number: {self.cam_number}")
            return

        # Test if camera can actually grab frames
        ret, test_frame = self.cam.read()
        if not ret:
            self.label.setText(f"Camera {self.cam_number} cannot grab frames!")
            self.cam.release()
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)  # 20 ms = 50 fps: 1000ms/FPS = ms per second
        self.is_running = True

    def update_frame(self):
        if not self.cam or not self.cam.isOpened():
            return

        ret, frame = self.cam.read()  # ret is boolean if frame is read correctly and frame is the image
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        height, width, channel = frame.shape
        bytesPerLine = channel * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)  # convert QImage to QPixmap
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def stop(self):
        """Stop the camera feed and release resources"""
        if self.timer:
            self.timer.stop()
        if self.cam and self.cam.isOpened():
            self.cam.release()
        self.is_running = False

def findcams(max_cams=3):
    """Find available cameras. Reduced default to 3 to avoid unnecessary scanning."""
    available_cams = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        if cap.isOpened():
            # Test if camera can actually grab frames
            ret, _ = cap.read()
            if ret:
                available_cams.append(i)
            cap.release()
    return available_cams