"""
### SignAI - Sign Language translater ###

## main.py for desktop app ##

-------------------------------------------------------------

Camera module for handling camera operations.

-------------------------------------------------------------

## Metadata
- **Author**: Stefanos Koufogazos Loukianov
- **Original Creation Date**: 2025/10/11
- **Last Update**: 2025/10/11

"""

import cv2
import os
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
import threading

# Suppress OpenCV warnings about camera indices
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

class Camera:
    def __init__(self, camera_id=0, resolution=(640, 480), fps=30):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.filepath = os.path.join(os.path.dirname(__file__), "..", "data", "live", "video", "recorded_video.mp4")

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        # setup camera
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Set buffer size to reduce lag
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.out = None
        self.recording = False
        self.thread = None

    def start_recording(self):
        if self.recording:
            return  # Already recording

        # Delete old video if exists
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except Exception as e:
                print(f"Could not delete old video: {e}")

        # Use MP4V codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.filepath, fourcc, self.fps, self.resolution)

        if not self.out or not self.out.isOpened():
            print(f"Error: Could not open VideoWriter with mp4v, trying MJPG")
            # Fallback to MJPG
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.filepath = self.filepath.replace('.mp4', '.avi')
            self.out = cv2.VideoWriter(self.filepath, fourcc, self.fps, self.resolution)

            if not self.out or not self.out.isOpened():
                print(f"Error: Could not open VideoWriter")
                return

        self.recording = True
        self.thread = threading.Thread(target=self._record_loop, daemon=False)
        self.thread.start()
        print(f"Recording started, saving to: {self.filepath}")

    def _record_loop(self):
        import time
        frame_delay = 1.0 / self.fps
        frame_count = 0

        while self.recording:
            start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.01)
                continue

            # Resize frame to match resolution
            if frame.shape[:2][::-1] != self.resolution:
                frame = cv2.resize(frame, self.resolution)

            self.out.write(frame)
            frame_count += 1

            # Maintain proper frame rate
            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

        print(f"Recording loop ended. Total frames: {frame_count}")

    def stop_recording(self):
        if not self.recording:
            return

        print("Stopping recording...")
        self.recording = False

        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)

        # Properly release the video writer
        if self.out:
            self.out.release()
            self.out = None
            # Give the system time to finalize the file
            import time
            time.sleep(0.5)

        print(f"Recording stopped and saved to: {self.filepath}")

        # Verify file was created and has size
        if os.path.exists(self.filepath):
            file_size = os.path.getsize(self.filepath)
            print(f"Video file size: {file_size} bytes")
            if file_size == 0:
                print("WARNING: Video file is empty!")
        else:
            print("WARNING: Video file was not created!")

    def save_video(self, filepath=None):
        # This method is now just for compatibility
        # The video is already saved when stop_recording() is called
        if filepath and filepath != self.filepath:
            # If a new filepath is specified, rename the file
            import shutil
            old_path = self.filepath
            new_path = os.path.join(os.path.dirname(old_path), filepath)
            if os.path.exists(old_path):
                try:
                    shutil.move(old_path, new_path)
                    self.filepath = new_path
                    print(f"Video moved to: {new_path}")
                except Exception as e:
                    print(f"Error moving video: {e}")

    def close(self):
        self.stop_recording()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

class CameraFeed:
    def __init__(self, label, cam_number=0):
        self.label = label
        self.cam_number = cam_number
        self.cam = cv2.VideoCapture(self.cam_number, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        self.timer = None
        self.is_running = False

        # Recording attributes
        self.recording = False
        self.out = None
        self.filepath = os.path.join(os.path.dirname(__file__), "..", "data", "live", "video", "recorded_video.mp4")
        self.fps = 30
        self.resolution = (640, 480)

        # Check if camera could get image
        if not self.cam.isOpened():
            self.label.setText(f"No Camera Found! Cam Number: {self.cam_number}")
            return

        # Set camera properties
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cam.set(cv2.CAP_PROP_FPS, self.fps)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Test if camera can actually grab frames
        ret, test_frame = self.cam.read()
        if not ret:
            self.label.setText(f"Camera {self.cam_number} cannot grab frames!")
            self.cam.release()
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # 33 ms = ~30 fps to match recording
        self.is_running = True

    def update_frame(self):
        if not self.cam or not self.cam.isOpened():
            return

        ret, frame = self.cam.read()  # ret is boolean if frame is read correctly and frame is the image
        if not ret:
            return

        # Resize frame to match resolution
        if frame.shape[:2][::-1] != self.resolution:
            frame = cv2.resize(frame, self.resolution)

        # If recording, write frame to video file
        if self.recording and self.out:
            self.out.write(frame)

        # Convert and display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        height, width, channel = frame_rgb.shape
        bytesPerLine = channel * width
        qImg = QImage(frame_rgb.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)  # convert QImage to QPixmap
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def start_recording(self):
        """Start recording while keeping the preview active"""
        if self.recording:
            return  # Already recording

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        # Delete old video if exists
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except Exception as e:
                print(f"Could not delete old video: {e}")

        # Use MP4V codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.filepath, fourcc, self.fps, self.resolution)

        if not self.out or not self.out.isOpened():
            print(f"Error: Could not open VideoWriter with mp4v, trying MJPG")
            # Fallback to MJPG
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.filepath = self.filepath.replace('.mp4', '.avi')
            self.out = cv2.VideoWriter(self.filepath, fourcc, self.fps, self.resolution)

            if not self.out or not self.out.isOpened():
                print(f"Error: Could not open VideoWriter")
                return False

        self.recording = True
        print(f"Recording started, saving to: {self.filepath}")
        return True

    def stop_recording(self):
        """Stop recording while keeping the preview active"""
        if not self.recording:
            return

        print("Stopping recording...")
        self.recording = False

        # Properly release the video writer
        if self.out:
            self.out.release()
            self.out = None
            # Give the system time to finalize the file
            import time
            time.sleep(0.5)

        print(f"Recording stopped and saved to: {self.filepath}")

        # Verify file was created and has size
        if os.path.exists(self.filepath):
            file_size = os.path.getsize(self.filepath)
            print(f"Video file size: {file_size} bytes")
            if file_size == 0:
                print("WARNING: Video file is empty!")
        else:
            print("WARNING: Video file was not created!")

    def stop(self):
        """Stop the camera feed and release resources"""
        # Stop recording if active
        if self.recording:
            self.stop_recording()

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
