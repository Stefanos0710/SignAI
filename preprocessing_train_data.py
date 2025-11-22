"""
SignAI - Sign Language Translator
Preprocessing Training Data Module

This script processes sign language videos to detect and track facial expressions,
hand gestures, and body poses using MediaPipe. It extracts keypoint data and saves
it to CSV files for model training.

Key Features:
- Face detection and landmark tracking
- Hand gesture recognition
- Body pose estimation
- Multi-frame processing with visualization
- Data normalization and scaling
- CSV output generation
- Enhanced UI and visualization
- Improved hand detection stability

Author: Stefanos Koufogazos Loukianov
Created: 2024-10-24
Updated: 2025-06-04 17:56:56

"""

import cv2
import mediapipe as mp
import csv
import os
import time as t
import numpy as np
import logging
import multiprocessing
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# settings for faster preprocessing
USE_GUI = False
USE_FACE = False
FRAME_SKIP = 2

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# UI Constants
WINDOW_TITLE = "SignAI - Sign Language Translator"
WINDOW_TITLES = {
    "main": "SignAI - Main View",
    "face": "Face Detection",
    "left_hand": "Left Hand Analysis",
    "right_hand": "Right Hand Analysis"
}

# UI Colors in BGR format
COLORS = {
    "primary": (0, 255, 0),    # Green
    "secondary": (255, 0, 0),  # Blue
    "accent": (0, 165, 255),   # Orange
    "text": (255, 255, 255),   # White
    "warning": (0, 0, 255)     # Red
}

# UI Layout
WINDOW_POSITIONS = {
    "main": (0, 0),
    "face": (900, 100),
    "left_hand": (1300, 100),
    "right_hand": (1300, 430)
}

WINDOW_SIZES = {
    "main": (1300, 800),
    "face": (400, 400),
    "hand": (300, 300)
}

# Detection Constants
FPS_TARGET = 20        # Target frames per second
FACE_ZOOM_SCALE = 2.0  # Face detection zoom factor
HAND_ZOOM_SCALE = 2.0  # Hand detection zoom factor
FACE_PADDING = 50      # Padding around face ROI
HAND_PADDING = 25      # Padding around hand ROI
HAND_BOX_SIZE = 70     # Size of hand detection box


class VideoProcessor:
    """
    Video processing class optimized for headless fast keypoint extraction.

    Key speedups applied:
    - GUI disabled (no namedWindow/imshow/waitKey) when use_gui=False
    - Face keypoints extraction disabled (USE_FACE=False) to reduce features to 150
    - MediaPipe Hands and Pose use tracking mode (static_image_mode=False) and low model_complexity
    - Optional frame skipping to drastically reduce processing (FRAME_SKIP)
    - Drawing and heavy visualization disabled when use_gui=False
    """

    def __init__(self, accuracy=0.3, rotation=0, use_gui=USE_GUI, use_face=USE_FACE, frame_skip=FRAME_SKIP):
        self.rotation = rotation
        self.use_gui = use_gui
        self.use_face = use_face
        self.frame_skip = max(1, int(frame_skip))

        # Hands: use tracking (static_image_mode=False) and low complexity for speed
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )

        # Pose: use lower complexity for speed
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize face modules only if requested (slower)
        if self.use_face:
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )

            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=accuracy,
                min_tracking_confidence=accuracy
            )
        else:
            self.face_detector = None
            self.face_mesh = None

        # Variables for scaling
        self.initial_scale_factor = None
        self.initial_shoulder_width = None

    def rotate_frame(self, frame):
        if self.rotation == 0:
            return frame
        elif self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return frame

    def setup_windows(self):
        # Only set up windows when GUI enabled
        if not self.use_gui:
            return
        for title, pos in WINDOW_POSITIONS.items():
            try:
                cv2.namedWindow(WINDOW_TITLES[title], cv2.WINDOW_NORMAL)
                cv2.moveWindow(WINDOW_TITLES[title], pos[0], pos[1])
                size = WINDOW_SIZES["main"] if title == "main" else WINDOW_SIZES["face"] if title == "face" else WINDOW_SIZES["hand"]
                cv2.resizeWindow(WINDOW_TITLES[title], size[0], size[1])
            except Exception as e:
                logging.debug(f"Could not create window {title}: {e}")

    def create_enhanced_ui(self, frame, detection_info, frame_count):
        # Skip heavy UI rendering when headless
        if not self.use_gui:
            return frame
        h, w = frame.shape[:2]
        ui_frame = frame.copy()

        # Create semi-transparent overlay for info panel
        overlay = ui_frame.copy()
        info_panel_height = 180
        cv2.rectangle(overlay, (0, 0), (250, info_panel_height),
                      (0, 0, 0), -1)

        # Add overlay with transparency
        alpha = 0.7
        ui_frame = cv2.addWeighted(overlay, alpha, ui_frame, 1 - alpha, 0)

        # Add title
        cv2.putText(ui_frame, "SignAI Analyzer",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, COLORS["accent"], 2)

        # Add frame information
        info_text = [
            f"Frame: {frame_count}",
            f"Hands: {detection_info.get('hands_detected', 0)}/2",
            f"Face: {'Detected' if detection_info.get('face_detected', False) else 'Not Detected'}",
            "",
            "Controls:",
            "SPACE - Next frame",
            "B - Previous frame",
            "ESC - Exit"
        ]

        for i, text in enumerate(info_text):
            y_pos = 60 + i * 20
            cv2.putText(ui_frame, text,
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, COLORS["text"], 1)

        # Add progress bar
        if detection_info.get('total_frames'):
            progress = frame_count / detection_info['total_frames']
            bar_width = 200
            bar_height = 15
            filled_width = int(bar_width * progress)

            # Draw progress bar background
            cv2.rectangle(ui_frame,
                          (w - bar_width - 20, 20),
                          (w - 20, 20 + bar_height),
                          COLORS["secondary"], -1)

            # Draw filled progress
            cv2.rectangle(ui_frame,
                          (w - bar_width - 20, 20),
                          (w - bar_width - 20 + filled_width, 20 + bar_height),
                          COLORS["accent"], -1)

            # Add percentage text
            percentage = f"{progress * 100:.1f}%"
            cv2.putText(ui_frame, percentage,
                        (w - bar_width - 20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, COLORS["text"], 1)

        return ui_frame

    def add_window_styling(self, frame, title, frame_count):
        if not self.use_gui:
            return frame
        h, w = frame.shape[:2]
        styled = frame.copy()

        # Add title bar
        cv2.rectangle(styled, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(styled, title,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, COLORS["text"], 2)

        # Add frame counter
        cv2.putText(styled, f"Frame: {frame_count}",
                    (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COLORS["text"], 1)

        return styled

    def calculate_initial_scale(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(image)

        if pose_results.pose_landmarks:
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            h, w = frame.shape[:2]
            left_px = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            right_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            shoulder_distance = np.sqrt((right_px[0] - left_px[0])**2 + (right_px[1] - left_px[1])**2)
            self.initial_scale_factor = max(shoulder_distance, 1.0)
            self.initial_shoulder_width = shoulder_distance
            logging.info(f"Initial scale factor set to: {self.initial_scale_factor}")
            return True

        logging.warning("Could not detect shoulders in first frame")
        return False

    def get_hands_roi(self, frame, pose_results):
        # Simplified: instead of heavy ROI zooms, rely on MediaPipe hands full-frame processing
        # This avoids repeated resizing which costs time
        return None, None

    def process_hand_keypoints(self, hand_landmarks):
        # hand_landmarks: mediapipe.landmark list
        hand_data = []
        # Assuming ref_x, ref_y, shoulder_width are provided externally
        # This helper can be used inlined in process_frame for speed
        return hand_data

    def process_frame(self, frame, frame_count):
        try:
            frame = self.rotate_frame(frame)

            if self.initial_scale_factor is None:
                if not self.calculate_initial_scale(frame):
                    return None, frame

            image = frame.copy()
            # Convert once
            mp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image.flags.writeable = False

            # Run pose and hands. Use tracking mode for speed.
            pose_results = self.pose.process(mp_image)
            hands_results = self.hands.process(mp_image)

            mp_image.flags.writeable = True

            if not pose_results.pose_landmarks:
                logging.warning(f"No pose detected in frame {frame_count}")
                return None, frame

            # Reference point
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            ref_x = (left_shoulder.x + right_shoulder.x) / 2
            ref_y = (left_shoulder.y + right_shoulder.y) / 2
            shoulder_width = np.sqrt((right_shoulder.x - left_shoulder.x) ** 2 + (right_shoulder.y - left_shoulder.y) ** 2)
            shoulder_width = max(shoulder_width, 1e-6)

            current_data = [frame_count]

            # Pose data (33 landmarks * 2)
            pose_data = []
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                x = (landmark.x - ref_x) / shoulder_width * 100
                y = (landmark.y - ref_y) / shoulder_width * 100
                pose_data.extend([x, y])

            # Hands data (2 hands * 21 * 2 = 84)
            hand_data = []
            if hands_results and hands_results.multi_hand_landmarks:
                # We will order hands by handedness if available
                # Collect up to two hands
                detected = []
                for hl in hands_results.multi_hand_landmarks:
                    coords = []
                    for lm in hl.landmark:
                        x = (lm.x - ref_x) / shoulder_width * 100
                        y = (lm.y - ref_y) / shoulder_width * 100
                        coords.extend([x, y])
                    detected.append(coords)

                # Flatten into left then right slot (if only one hand, fill remaining with zeros)
                # We don't do complex side detection here to keep it fast
                for coords in detected[:2]:
                    hand_data.extend(coords)

            # Pad missing hand data to 84 values
            if len(hand_data) < 42 * 2:
                hand_data.extend([0.0] * (42 * 2 - len(hand_data)))

            # Face data skipped when use_face==False
            # Combine
            current_data.extend(pose_data)
            current_data.extend(hand_data)

            # Minimal visualization only when GUI enabled
            processed_frame = image if self.use_gui else image

            return current_data, processed_frame

        except Exception as e:
            logging.error(f"Error in frame {frame_count}: {str(e)}")
            return None, frame

    def process_video(self, video_path, output_csv_path, video_name, gloss_text, fps):
        """Streaming frame-by-frame processing without caching frames to memory. This is faster and uses less RAM.
        """
        cam = cv2.VideoCapture(video_path)
        if not cam.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return False

        # Only setup windows if GUI enabled
        if self.use_gui:
            self.setup_windows()

        original_fps = int(cam.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.total_frames = total_frames

        logging.info(f"Original video FPS: {original_fps}")
        logging.info(f"Total frames: {total_frames}")

        # compute processing stride to match target fps and frame_skip
        try:
            stride = max(1, int(round(original_fps / max(1, fps))))
        except Exception:
            stride = 1
        stride = max(1, stride * self.frame_skip)

        # Open CSV file
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            headers = ["Video_Name", "Gloss", "Frame"]
            headers += [f"pose_{i}_{axis}" for i in range(33) for axis in ['x', 'y']]
            headers += [f"hand_{i}_{axis}" for i in range(42) for axis in ['x', 'y']]
            writer.writerow(headers)

            processed_frames = 0
            failed_frames = 0

            frame_idx = 0
            read_idx = 0
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                read_idx += 1
                # skip frames according to stride
                if (read_idx - 1) % stride != 0:
                    continue

                frame_idx += 1
                current_data, _ = self.process_frame(frame, read_idx)

                if current_data is not None:
                    writer.writerow([video_name, gloss_text] + current_data)
                    processed_frames += 1
                else:
                    failed_frames += 1

                # Minimal GUI handling
                if self.use_gui:
                    try:
                        cv2.imshow(WINDOW_TITLES['main'], frame)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                    except Exception:
                        pass

        cam.release()

        logging.info("\nVideo processing completed:")
        logging.info(f"Total frames: {total_frames}")
        logging.info(f"Processed frames: {processed_frames}")
        logging.info(f"Failed frames: {failed_frames}")
        logging.info(f"Success rate: {(processed_frames / max(1, (total_frames or 1))) * 100:.2f}%")

        if self.use_gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        return True


def get_video_orientation():
    while True:
        print("\nVideo Orientation for ALL videos:")
        print("1. Normal (Horizontal)")
        print("2. Rotated Right (Portrait - Phone held in right hand)")
        print("3. Rotated Left (Portrait - Phone held in left hand)")
        choice = input("Enter video orientation (1-3): ")
        if choice == "1":
            return 0
        elif choice == "2":
            return 90
        elif choice == "3":
            return 270
        else:
            print("Please enter a valid choice (1-3)")


def get_fps_from_user():
    """Return FPS to use. In non-interactive or headless mode returns FPS_TARGET.
    Allows overriding via environment variable SIGNAI_FPS.
    """
    # 1) environment override
    env = os.environ.get('SIGNAI_FPS')
    if env:
        try:
            fps = float(env)
            if 1 <= fps <= 240:
                logging.info(f"Using SIGNAI_FPS from environment: {fps}")
                return fps
            else:
                logging.warning(f"SIGNAI_FPS out of range (1-240): {env}. Falling back to default {FPS_TARGET}.")
        except Exception:
            logging.warning(f"Invalid SIGNAI_FPS value: {env}. Falling back to default {FPS_TARGET}.")

    # 2) if running non-interactively or GUI disabled, return default
    try:
        if not sys.stdin.isatty() or not USE_GUI:
            logging.info(f"Non-interactive or headless mode detected, using default FPS: {FPS_TARGET}")
            return FPS_TARGET
    except Exception:
        # If anything goes wrong, fallback to default
        return FPS_TARGET

    # 3) interactive prompt fallback
    while True:
        try:
            print("\nPlayback Speed Options:")
            print("1. Normal speed (30 FPS)")
            print("2. Fast speed (60 FPS)")
            print("3. Very fast speed (120 FPS)")
            print("4. Custom FPS")
            choice = input("Enter your choice (1-4): ")
            if choice == "1":
                return 30
            elif choice == "2":
                return 60
            elif choice == "3":
                return 120
            elif choice == "4":
                custom_fps = input("Enter custom FPS (1-240): ")
                fps = float(custom_fps)
                if 1 <= fps <= 240:
                    return fps
                print("Please enter a value between 1 and 240")
            else:
                print("Please enter a valid choice (1-4)")
        except ValueError:
            print("Please enter a valid number")


def process_single_video(task):
    """Top-level worker function for multiprocessing.
    task: dict with keys: video_file, csv_file_path, subfolder_name, gloss_text, fps, rotation, use_face, frame_skip
    """
    try:
        video_file = task['video_file']
        csv_file_path = task['csv_file_path']
        subfolder_name = task['subfolder_name']
        gloss_text = task['gloss_text']
        fps = task['fps']
        rotation = task['rotation']
        use_face = task.get('use_face', False)
        frame_skip = task.get('frame_skip', FRAME_SKIP)

        # Create a local processor inside the worker (MediaPipe objects are not picklable)
        processor = VideoProcessor(accuracy=0.3, rotation=rotation, use_gui=False, use_face=use_face, frame_skip=frame_skip)
        success = processor.process_video(video_file, csv_file_path, subfolder_name, gloss_text, fps)
        return (subfolder_name, success, None)
    except Exception as e:
        return (task.get('subfolder_name', 'unknown'), False, str(e))


def main():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        input_videos_folder = os.path.join(base_dir, "data", "raw_data")
        output_videos_folder = os.path.join(base_dir, "data", "train_data")

        try:
            os.makedirs(output_videos_folder, exist_ok=True)
            logging.info(f"Output directory created/verified: {output_videos_folder}")
        except Exception as e:
            logging.error(f"Failed to create output directory: {str(e)}")
            return

        if not os.path.exists(input_videos_folder):
            logging.error(f"Input directory does not exist: {input_videos_folder}")
            return

        fps = get_fps_from_user()
        print(f"\nSelected playback speed: {fps} FPS")
        rotation = get_video_orientation()
        print(f"\nSelected rotation for all videos: {rotation}°")

        # Build list of tasks (one per video folder)
        tasks = []
        for subfolder_name in os.listdir(input_videos_folder):
            subfolder_path = os.path.join(input_videos_folder, subfolder_name)
            if not os.path.isdir(subfolder_path):
                continue

            video_file = None
            gloss_file = None
            for file_name in os.listdir(subfolder_path):
                if file_name.lower().endswith('.mp4'):
                    video_file = os.path.join(subfolder_path, file_name)
                elif file_name.lower().endswith('.txt'):
                    gloss_file = os.path.join(subfolder_path, file_name)

            if not video_file or not gloss_file:
                logging.warning(f"Missing files in folder: {subfolder_name}")
                continue

            try:
                with open(gloss_file, 'r', encoding='utf-8') as f:
                    gloss_text = f.read().strip()
            except Exception as e:
                logging.error(f"Failed to read gloss in {subfolder_name}: {e}")
                continue

            csv_file_path = os.path.join(output_videos_folder, f"{subfolder_name}_traindata.csv")
            # Pre-check file creation
            try:
                with open(csv_file_path, 'w', encoding='utf-8'):
                    pass
            except Exception as e:
                logging.error(f"Cannot create csv file {csv_file_path}: {e}")
                continue

            tasks.append({
                'video_file': video_file,
                'csv_file_path': csv_file_path,
                'subfolder_name': subfolder_name,
                'gloss_text': gloss_text,
                'fps': fps,
                'rotation': rotation,
                'use_face': USE_FACE,
                'frame_skip': FRAME_SKIP,
            })

        if not tasks:
            logging.info("No tasks to process.")
            return

        # Determine worker count: use CPU count but leave one core free
        workers = max(1, multiprocessing.cpu_count() - 1)
        logging.info(f"Starting multiprocessing pool with {workers} workers to process {len(tasks)} videos")

        # Use multiprocessing Pool to process videos in parallel
        with multiprocessing.Pool(processes=workers) as pool:
            for subfolder_name, success, err in pool.imap_unordered(process_single_video, tasks):
                if success:
                    logging.info(f"Processed {subfolder_name} successfully")
                else:
                    logging.error(f"Failed to process {subfolder_name}: {err}")

    except Exception as e:
        logging.error(f"Main execution error: {str(e)}")
    finally:
        if USE_GUI:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


# Ensure script entry-point safe for multiprocessing on Windows
if __name__ == '__main__':
    try:
        main()
    finally:
        if USE_GUI:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
