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
from videos import HistoryVideos

# add writable_path import
from resource_path import writable_path

# Suppress OpenCV warnings about camera indices
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

# Preferred backends order (DSHOW -> MSMF -> ANY) with graceful fallback
BACKENDS = [
    getattr(cv2, "CAP_DSHOW", 700),
    getattr(cv2, "CAP_MSMF", 1400),
    getattr(cv2, "CAP_ANY", 0),
]


def _open_capture(cam_number: int):
    """Try to open a camera using a list of backends, returning the first working VideoCapture."""
    for be in BACKENDS:
        try:
            cap = cv2.VideoCapture(cam_number, be)
            if cap and cap.isOpened():
                return cap
            if cap:
                cap.release()
        except Exception as e:
            print(f"[Camera] backend {be} failed for cam {cam_number}: {e}")
    return None

class Camera:
    def __init__(self, camera_id=0, resolution=(640, 480), fps=30, camera_feed=None):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        # Use a writable path for runtime files so bundling doesn't write into the package
        self.filepath = writable_path(os.path.join("videos", "current_video.mp4"))
        self.camera_feed = camera_feed  # Reference to CameraFeed to share frames

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        # Don't open camera in __init__, wait for start_recording
        self.cap = None
        self.out = None
        self.recording = False
        self.thread = None
        self.latest_frame = None  # Store latest frame for sharing

    def start_recording(self):
        import time

        if self.recording:
            return  # Already recording

        # Pause the camera feed if it exists
        if self.camera_feed:
            self.camera_feed.pause()

        # Wait a bit to ensure previous camera release is complete
        time.sleep(0.3)

        # Setup camera
        self.cap = _open_capture(self.camera_id)

        if not self.cap or not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            if self.camera_feed:
                self.camera_feed.resume()
            return

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Test if we can grab frames
        for attempt in range(5):
            ret, test_frame = self.cap.read()
            if ret:
                break
            print(f"Warming up camera, attempt {attempt + 1}/5...")
            time.sleep(0.2)

        if not ret:
            print("Error: Camera cannot grab frames after warmup")
            self.cap.release()
            self.cap = None
            if self.camera_feed:
                self.camera_feed.resume()
            return

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
                if self.cap:
                    self.cap.release()
                    self.cap = None
                if self.camera_feed:
                    self.camera_feed.resume()
                return

        self.recording = True
        self.thread = threading.Thread(target=self._record_loop, daemon=False)
        self.thread.start()
        print(f"Recording started, saving to: {self.filepath}")

    def _record_loop(self):
        import time
        frame_delay = 1.0 / self.fps
        frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 30  # Stop after 30 consecutive failures

        while self.recording:
            start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"Too many consecutive failures ({consecutive_failures}), stopping recording")
                    self.recording = False
                    break
                print(f"Failed to grab frame (failure {consecutive_failures}/{max_consecutive_failures})")
                time.sleep(0.05)
                continue

            # Reset failure counter on success
            consecutive_failures = 0

            # Resize frame to match resolution
            if frame.shape[:2][::-1] != self.resolution:
                frame = cv2.resize(frame, self.resolution)

            # Store latest frame for camera feed
            self.latest_frame = frame.copy()

            # Update camera feed if available
            if self.camera_feed:
                self.camera_feed.update_from_recording(frame)

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

        # Release camera
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        # Give the system time to finalize the file
        import time
        time.sleep(0.5)

        print(f"Recording stopped and saved to: {self.filepath}")

        # Resume camera feed
        if self.camera_feed:
            self.camera_feed.resume()

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
            self.cap = None
        cv2.destroyAllWindows()

class CameraFeed:
    def __init__(self, label, cam_number=0):
        self.label = label
        self.cam_number = cam_number
        self.cam = _open_capture(self.cam_number)
        self.timer = None
        self.is_running = False
        self.is_paused = False

        # Remove recording attributes - CameraFeed is only for display
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

        # Warm up camera by reading several frames
        import time
        ret = False
        for attempt in range(10):
            ret, test_frame = self.cam.read()
            if ret:
                break
            time.sleep(0.1)

        if not ret:
            self.label.setText(f"Camera {self.cam_number} cannot grab frames!")
            self.cam.release()
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # 33 ms = ~30 fps
        self.is_running = True

    def update_frame(self):
        if not self.cam or not self.cam.isOpened() or self.is_paused:
            return

        ret, frame = self.cam.read()  # ret is boolean if frame is read correctly and frame is the image
        if not ret:
            return

        self._display_frame(frame)

    def update_from_recording(self, frame):
        """Update display from recording frame (called from Camera class)"""
        self._display_frame(frame)

    def _display_frame(self, frame):
        """Internal method to display a frame"""
        # Resize frame to match resolution
        if frame.shape[:2][::-1] != self.resolution:
            frame = cv2.resize(frame, self.resolution)

        # Convert and display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        height, width, channel = frame_rgb.shape
        bytesPerLine = channel * width
        qImg = QImage(frame_rgb.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)  # convert QImage to QPixmap
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def pause(self):
        """Pause the camera feed without releasing the camera"""
        self.is_paused = True
        if self.timer:
            self.timer.stop()
        if self.cam and self.cam.isOpened():
            self.cam.release()
        print(f"Camera feed paused for camera {self.cam_number}")

    def resume(self):
        """Resume the camera feed"""
        import time

        self.is_paused = False

        # Wait a bit before reopening
        time.sleep(0.3)

        # Reopen camera
        self.cam = _open_capture(self.cam_number)
        if not self.cam or not self.cam.isOpened():
            self.label.setText(f"Error: Cannot reopen camera {self.cam_number}")
            return

        # Set camera properties
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cam.set(cv2.CAP_PROP_FPS, self.fps)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm up camera by reading several frames
        for attempt in range(5):
            ret, _ = self.cam.read()
            if ret:
                break
            time.sleep(0.1)

        # Restart timer
        if self.timer:
            self.timer.start(33)
        print(f"Camera feed resumed for camera {self.cam_number}")

    def stop(self):
        """Stop the camera feed and release resources"""
        if self.timer:
            self.timer.stop()
        if self.cam and self.cam.isOpened():
            self.cam.release()
        self.is_running = False
        self.is_paused = False

def _probe_camera(index: int, backend, timeout_sec: float = 1.5) -> bool:
    """Probe a camera index with a backend; return True if a frame can be read quickly."""
    import time
    cap = cv2.VideoCapture(index, backend)
    if not cap or not cap.isOpened():
        return False
    start = time.time()
    ret = False
    for _ in range(5):
        ret, _frame = cap.read()
        if ret:
            break
        if time.time() - start > timeout_sec:
            break
        time.sleep(0.1)
    cap.release()
    return bool(ret)


def findcams(max_cams=3):
    """Find available cameras quickly without hanging on bad drivers."""
    available_cams = []
    for i in range(max_cams):
        found = False
        for be in BACKENDS:
            try:
                if _probe_camera(i, be):
                    available_cams.append(i)
                    found = True
                    break
            except Exception as e:
                print(f"[Camera] probe failed for index {i} backend {be}: {e}")
        if not found:
            print(f"[Camera] index {i} not usable")
    return available_cams
