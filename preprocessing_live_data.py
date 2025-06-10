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
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    A comprehensive video processing class for sign language detection.

    This class handles all aspects of video analysis including:
    - Face detection and landmark tracking (468 points)
    - Hand gesture recognition (21 points per hand)
    - Body pose estimation (33 points)
    - Frame enhancement and preprocessing
    - Data normalization and scaling
    - Enhanced UI visualization
    - Improved hand detection stability

    Attributes:
        hands: MediaPipe hands solution instance
        face_detector: MediaPipe face detection instance
        face_mesh: MediaPipe face mesh instance
        pose: MediaPipe pose detection instance
        initial_scale_factor: Scaling factor based on shoulder width
        initial_shoulder_width: Reference shoulder width for scaling
    """

    def __init__(self, accuracy=0.3):
        """
        Initialize the VideoProcessor with detection models and parameters.

        Args:
            accuracy (float): Base accuracy threshold for detections (0.0-1.0)
        """
        # Hand detection with optimized settings
        self.hands = mp_hands.Hands(
            static_image_mode=True,  # Better frame-by-frame detection
            max_num_hands=2,
            min_detection_confidence=0.05,  # Lower threshold for better detection
            min_tracking_confidence=0.05,
            model_complexity=1  # Balanced complexity
        )

        # Initialize face detection for rough localization
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        # Initialize face mesh for detailed facial landmarks
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=accuracy,
            min_tracking_confidence=accuracy
        )

        # Initialize pose detection
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=accuracy,
            min_tracking_confidence=accuracy
        )

        # Variables for scaling
        self.initial_scale_factor = None
        self.initial_shoulder_width = None

    def setup_windows(self):
        """
        Sets up all visualization windows with proper sizes and positions.
        Ensures consistent window layout across sessions.
        """
        for title, pos in WINDOW_POSITIONS.items():
            cv2.namedWindow(WINDOW_TITLES[title], cv2.WINDOW_NORMAL)
            cv2.moveWindow(WINDOW_TITLES[title], pos[0], pos[1])

            # Set window size based on type
            size = WINDOW_SIZES["main"] if title == "main" else \
                WINDOW_SIZES["face"] if title == "face" else \
                    WINDOW_SIZES["hand"]
            cv2.resizeWindow(WINDOW_TITLES[title], size[0], size[1])

    def create_enhanced_ui(self, frame, detection_info, frame_count):
        """
        Creates an enhanced UI overlay for the frame visualization.

        Args:
            frame (np.ndarray): The input frame to add UI elements to
            detection_info (dict): Dictionary containing detection information
            frame_count (int): Current frame number

        Returns:
            np.ndarray: Frame with enhanced UI elements
        """
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
        """
        Adds consistent styling to detection windows.

        Args:
            frame (np.ndarray): Input frame
            title (str): Window title
            frame_count (int): Current frame number

        Returns:
            np.ndarray: Styled frame
        """
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
        """
        Calculate initial scaling factor based on shoulder width in the first frame.

        Args:
            frame (np.ndarray): First video frame in BGR format

        Returns:
            bool: True if scale calculation successful, False otherwise
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(image)

        if pose_results.pose_landmarks:
            # Get shoulder landmarks
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Convert to pixel coordinates
            h, w = frame.shape[:2]
            left_px = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            right_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))

            # Calculate shoulder distance
            shoulder_distance = np.sqrt(
                (right_px[0] - left_px[0])**2 +
                (right_px[1] - left_px[1])**2
            )

            self.initial_scale_factor = max(shoulder_distance, 1.0)
            self.initial_shoulder_width = shoulder_distance
            logging.info(f"Initial scale factor set to: {self.initial_scale_factor}")
            return True

        logging.error("Could not detect shoulders in first frame")
        return False

    def get_face_roi(self, frame):
        """
        Extracts the Region of Interest (ROI) containing the face.

        Args:
            frame (np.ndarray): Input frame in BGR format

        Returns:
            tuple: (face_roi, coordinates) or (None, None) if no face detected
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(image)

        if not results.detections:
            return None, None

        # Use first detected face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        # Convert relative coordinates to absolute pixels
        h, w = frame.shape[:2]
        x_min = max(0, int(bbox.xmin * w))
        y_min = max(0, int(bbox.ymin * h))
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # Add padding around face
        padding = FACE_PADDING
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_min + width + 2 * padding)
        y_max = min(h, y_min + height + 2 * padding)

        # Extract face region
        face_roi = frame[y_min:y_max, x_min:x_max].copy()

        if face_roi.size > 0:
            # Resize for better detail
            new_width = int((x_max - x_min) * FACE_ZOOM_SCALE)
            new_height = int((y_max - y_min) * FACE_ZOOM_SCALE)
            face_roi = cv2.resize(face_roi, (new_width, new_height))

            return face_roi, (x_min, y_min, x_max, y_max)

        return None, None

    def get_hands_roi(self, frame, pose_results):
        """
        Extracts hand regions based on pose keypoints detection with dynamic scaling.
        Uses shoulder width to determine appropriate zoom level.
        """
        if not pose_results.pose_landmarks:
            return None, None

        h, w = frame.shape[:2]

        # Get shoulder landmarks for scaling calculation
        left_shoulder = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate shoulder width in pixels
        shoulder_width = int(abs(right_shoulder.x - left_shoulder.x) * w)

        # Dynamic scaling calculation
        # Base scale is 2.0, adjust based on shoulder width
        # If person is far (small shoulder width), increase zoom
        # If person is close (large shoulder width), decrease zoom
        base_scale = 2.0
        reference_shoulder_width = w * 0.08  # Expected shoulder width (30% of frame width)
        dynamic_scale = base_scale * (reference_shoulder_width / max(shoulder_width, 1))

        # Clamp scale between reasonable limits
        dynamic_scale = max(1.2, min(3.0, dynamic_scale))

        # Calculate dynamic box size based on shoulder width
        base_box_size = 80
        dynamic_box_size = int(base_box_size * (shoulder_width / reference_shoulder_width))
        # Clamp box size between reasonable limits
        dynamic_box_size = max(50, min(200, dynamic_box_size))

        logging.info(
            f"Shoulder width: {shoulder_width}px, Dynamic scale: {dynamic_scale:.2f}x, Box size: {dynamic_box_size}px")

        # Define wrist landmarks for both hands
        wrist_landmarks = {
            'Left': mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            'Right': mp.solutions.pose.PoseLandmark.RIGHT_WRIST
        }

        hands_roi = []
        roi_coords = []

        for side, landmark in wrist_landmarks.items():
            wrist = pose_results.pose_landmarks.landmark[landmark]

            # Check if wrist is visible
            if wrist.visibility > 0.5:
                # Center of hand (wrist position)
                center_x = int(wrist.x * w)
                center_y = int(wrist.y * h)

                # Create box around wrist with dynamic size
                half_box = dynamic_box_size // 2
                x_min = max(0, center_x - half_box)
                y_min = max(0, center_y - half_box)
                x_max = min(w, center_x + half_box)
                y_max = min(h, center_y + half_box)

                # Add padding proportional to box size
                padding = int(dynamic_box_size * 0.1)  # 10% padding
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                # Extract hand ROI
                hand_roi = frame[y_min:y_max, x_min:x_max].copy()

                if hand_roi.size > 0:
                    # Apply dynamic zoom
                    new_width = int((x_max - x_min) * dynamic_scale)
                    new_height = int((y_max - y_min) * dynamic_scale)

                    try:
                        hand_roi = cv2.resize(hand_roi, (new_width, new_height))

                        # Add debug visualization
                        cv2.putText(hand_roi,
                                    f"Scale: {dynamic_scale:.1f}x",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1)

                        hands_roi.append((side, hand_roi))
                        roi_coords.append((side, (x_min, y_min, x_max, y_max)))
                    except Exception as e:
                        logging.warning(f"Failed to resize {side} hand ROI: {str(e)}")
                        continue

        return hands_roi, roi_coords

    def process_face_keypoints(self, face_roi):
        """
        Processes facial landmarks using MediaPipe Face Mesh.

        Args:
            face_roi (np.ndarray): Face region image

        Returns:
            tuple: (landmarks, visualization) or (None, None) if no face detected
        """
        if face_roi is None:
            return None, None

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(face_rgb)

        if not face_results.multi_face_landmarks:
            return None, None

        # Create visualization
        face_viz = face_roi.copy()
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw mesh tesselation
            mp_drawing.draw_landmarks(
                image=face_viz,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(80, 110, 10),
                    thickness=1,
                    circle_radius=1
                )
            )

            # Draw contours
            mp_drawing.draw_landmarks(
                image=face_viz,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(80, 256, 121),
                    thickness=1,
                    circle_radius=1
                )
            )

        return face_results.multi_face_landmarks[0], face_viz

    def process_hand_keypoints(self, hand_roi, target_side):
        """
        Processes hand landmarks with corrected side detection.

        Args:
            hand_roi (np.ndarray): Hand region image
            target_side (str): Expected hand side ('Left' or 'Right')

        Returns:
            tuple: (landmarks, visualization) or (None, None) if no hand detected
        """
        if hand_roi is None:
            return None, None

        # MediaPipe processing
        hand_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(hand_rgb)

        if not hand_results.multi_hand_landmarks:
            return None, None

        # Visualization
        hand_viz = hand_roi.copy()
        detected_hand = None

        for idx, (hand_landmarks, handedness) in enumerate(
                zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):

            # Correct hand side detection (mirror image perspective)
            detected_side = "Right" if handedness.classification[0].label == "Left" else "Left"

            if detected_side == target_side:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image=hand_viz,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(121, 22, 76), thickness=2, circle_radius=4
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(121, 44, 250), thickness=2, circle_radius=2
                    )
                )
                detected_hand = hand_landmarks

                # Add detection confidence
                confidence = handedness.classification[0].score
                cv2.putText(hand_viz,
                           f"Confidence: {confidence:.2f}",
                           (10, hand_viz.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           COLORS["primary"],
                           1)
                break

        return detected_hand, hand_viz

    def process_frame(self, frame, frame_count):
        """
        Process a single video frame with enhanced detection and visualization.
        Normalizes coordinates relative to shoulder center point (0,0).

        Args:
            frame (np.ndarray): Input video frame in BGR format
            frame_count (int): Current frame number in sequence

        Returns:
            tuple: (data_array, processed_image) or (None, frame) if processing fails
        """
        try:
            # Create a copy of the frame for processing
            image = frame.copy()

            # Convert to RGB for MediaPipe
            mp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image.flags.writeable = False

            # Process pose
            pose_results = self.pose.process(mp_image)

            if not pose_results.pose_landmarks:
                logging.warning(f"No pose detected in frame {frame_count}")
                return None, frame

            mp_image.flags.writeable = True
            image_bgr = cv2.cvtColor(mp_image, cv2.COLOR_RGB2BGR)

            # Get shoulder landmarks for reference point calculation
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Calculate reference point (0,0) as center between shoulders
            ref_x = (left_shoulder.x + right_shoulder.x) / 2
            ref_y = (left_shoulder.y + right_shoulder.y) / 2

            # Calculate shoulder width for scaling
            shoulder_width = np.sqrt(
                (right_shoulder.x - left_shoulder.x) ** 2 +
                (right_shoulder.y - left_shoulder.y) ** 2
            )

            # Process face
            face_roi, face_coords = self.get_face_roi(image)
            face_landmarks = None
            face_viz = None

            if face_roi is not None:
                face_landmarks, face_viz = self.process_face_keypoints(face_roi)
                if face_viz is not None:
                    face_viz = self.add_window_styling(face_viz, "Face Analysis", frame_count)
                    cv2.imshow(WINDOW_TITLES["face"], face_viz)

            # Process hands
            hands_roi, roi_coords = self.get_hands_roi(image, pose_results)
            hand_landmarks_list = []
            hand_vizs = []

            if hands_roi:
                for (side, hand_roi), (_, coords) in zip(hands_roi, roi_coords):
                    hand_landmarks, hand_viz = self.process_hand_keypoints(hand_roi, side)
                    if hand_viz is not None:
                        hand_viz = self.add_window_styling(hand_viz, f"{side} Hand", frame_count)
                        cv2.imshow(WINDOW_TITLES[f"{side.lower()}_hand"], hand_viz)
                        hand_vizs.append(hand_viz)
                        if hand_landmarks:
                            hand_landmarks_list.append((side, hand_landmarks))

            # Prepare detection info for UI
            detection_info = {
                'hands_detected': len(hand_landmarks_list),
                'face_detected': face_landmarks is not None,
                'current_frame': frame_count
            }

            # Initialize data collection with frame number
            current_data = [frame_count]

            # Process pose data with normalization
            pose_data = []
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLORS["accent"], thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=COLORS["secondary"], thickness=2, circle_radius=2)
                )

                # Process pose landmarks relative to shoulder center
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    # Normalize coordinates relative to shoulder center and scale by shoulder width
                    x = (landmark.x - ref_x) / shoulder_width * 100
                    y = (landmark.y - ref_y) / shoulder_width * 100
                    pose_data.extend([x, y])

                    # Visualize important landmarks
                    x_px = int(landmark.x * image.shape[1])
                    y_px = int(landmark.y * image.shape[0])

                    if idx in [0, 11, 12, 15, 16]:  # Important landmarks
                        cv2.circle(image, (x_px, y_px), 5, COLORS["accent"], -1)
                        cv2.putText(image, f"P{idx}", (x_px, y_px - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["accent"], 1)
            else:
                pose_data = [0] * (33 * 2)

            # Process hand data with normalization
            hand_data = []
            for side, landmarks in hand_landmarks_list:
                for landmark in landmarks.landmark:
                    # Normalize coordinates relative to shoulder center and scale by shoulder width
                    x = (landmark.x - ref_x) / shoulder_width * 100
                    y = (landmark.y - ref_y) / shoulder_width * 100
                    hand_data.extend([x, y])

            # Fill missing hand data
            hand_data.extend([0] * (42 * 2 - len(hand_data)))

            # Process face data with normalization
            face_data = []
            if face_landmarks:
                for landmark in face_landmarks.landmark:
                    # Normalize coordinates relative to shoulder center and scale by shoulder width
                    x = (landmark.x - ref_x) / shoulder_width * 100
                    y = (landmark.y - ref_y) / shoulder_width * 100
                    face_data.extend([x, y])
            else:
                face_data = [0] * (468 * 2)

            # Create enhanced UI
            enhanced_frame = self.create_enhanced_ui(image, detection_info, frame_count)

            # Visualize reference point (shoulder center)
            center_x = int(ref_x * image.shape[1])
            center_y = int(ref_y * image.shape[0])
            cv2.circle(enhanced_frame, (center_x, center_y), 8, (0, 255, 255), -1)
            cv2.putText(enhanced_frame, "Reference (0,0)", (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Visualize shoulder width
            left_x = int(left_shoulder.x * image.shape[1])
            left_y = int(left_shoulder.y * image.shape[0])
            right_x = int(right_shoulder.x * image.shape[1])
            right_y = int(right_shoulder.y * image.shape[0])
            cv2.line(enhanced_frame, (left_x, left_y), (right_x, right_y), (0, 255, 255), 2)
            cv2.putText(enhanced_frame, f"Scale: {shoulder_width:.2f}",
                        ((left_x + right_x) // 2, left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Combine all data
            current_data.extend(pose_data)
            current_data.extend(hand_data)
            current_data.extend(face_data)

            return current_data, enhanced_frame

        except Exception as e:
            logging.error(f"Error in frame {frame_count}: {str(e)}")
            return None, frame

    def process_video(self, video_path, output_csv_path, video_name, gloss_text, fps):
        """
        Process complete video and save extracted keypoint data to CSV.
        Enhanced with high-FPS support and improved playback control.

        Args:
            video_path (str): Path to input video file
            output_csv_path (str): Path for output CSV file
            video_name (str): Name identifier for the video
            gloss_text (str): Associated sign language gloss text

        Returns:
            bool: True if processing successful, False otherwise
        """
        cam = cv2.VideoCapture(video_path)
        if not cam.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return False

        # Setup windows with enhanced styling
        self.setup_windows()

        # Get video properties
        original_fps = int(cam.get(cv2.CAP_PROP_FPS))
        total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = total_frames

        logging.info(f"Original video FPS: {original_fps}")
        logging.info(f"Total frames: {total_frames}")

        processed_frames = 0
        failed_frames = 0

        # Read all frames
        logging.info("Reading all frames...")
        frames = []
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            frames.append(frame)

        logging.info(f"Read {len(frames)} frames")

        # Calculate delay between frames in microseconds for higher precision
        frame_delay = int(1000000 / fps)  # microseconds
        min_delay = 1000  # 1ms minimum delay to prevent system overload
        mode = "auto"

        # Open CSV file and write headers
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Create headers
            headers = ["Video_Name", "Gloss", "Frame"]
            headers += [f"pose_{i}_{axis}" for i in range(33) for axis in ['x', 'y']]
            headers += [f"hand_{i}_{axis}" for i in range(42) for axis in ['x', 'y']]
            headers += [f"face_{i}_{axis}" for i in range(468) for axis in ['x', 'y']]

            writer.writerow(headers)

            # Process frames with high-precision timing
            frame_idx = 0
            is_paused = False
            last_frame_time = t.time()

            while frame_idx < len(frames):
                try:
                    current_time = t.time()

                    # Process current frame
                    current_data, processed_frame = self.process_frame(frames[frame_idx], frame_idx + 1)

                    # Show progress
                    progress = ((frame_idx + 1) / total_frames) * 100
                    logging.info(f"Processing frame {frame_idx + 1}/{total_frames} ({progress:.1f}%)")

                    if current_data is not None:
                        writer.writerow([video_name, gloss_text] + current_data)
                        processed_frames += 1
                    else:
                        failed_frames += 1
                        logging.warning(f"Frame {frame_idx + 1} processing failed")

                    # Add playback control info to the frame
                    control_info = [
                        f"Mode: {'Auto' if mode == 'auto' else 'Manual'}",
                        f"Speed: {fps:.1f} FPS",
                        f"Original FPS: {original_fps}",
                        "Controls:",
                        "SPACE - Play/Pause",
                        "M - Switch Mode",
                        "+ - Speed Up (10% faster)",
                        "- - Speed Down (10% slower)",
                        "F - Reset to original FPS",
                        "B - Previous Frame",
                        "ESC - Exit"
                    ]

                    # Add control info to the frame with improved visibility
                    overlay = processed_frame.copy()
                    cv2.rectangle(overlay, (5, processed_frame.shape[0] - 200),
                                  (250, processed_frame.shape[0] - 5), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)

                    for i, text in enumerate(control_info):
                        cv2.putText(processed_frame, text,
                                    (10, processed_frame.shape[0] - 180 + i * 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)

                    # Show main frame
                    cv2.imshow(WINDOW_TITLES["main"], processed_frame)

                    # Timing control for accurate FPS
                    if mode == 'auto' and not is_paused:
                        elapsed = (t.time() - last_frame_time) * 1000000  # microseconds
                        wait_time = max(min_delay, int(frame_delay - elapsed))
                        key = cv2.waitKey(1)  # Use minimum wait for high FPS

                        # Sleep for remaining time if needed
                        if wait_time > min_delay:
                            t.sleep(wait_time / 1000000.0)  # Convert to seconds
                    else:
                        key = cv2.waitKey(0)

                    if key != -1:  # If a key was pressed
                        if key == 27:  # ESC
                            break
                        elif key == 32:  # SPACE
                            is_paused = not is_paused
                        elif key == ord('m'):  # Switch mode
                            mode = 'manual' if mode == 'auto' else 'auto'
                        elif key == ord('b'):  # Previous frame
                            frame_idx = max(0, frame_idx - 2)
                        elif key == ord('+'):  # Speed up by 10%
                            fps = min(240, fps * 1.1)
                            frame_delay = int(1000000 / fps)
                        elif key == ord('-'):  # Slow down by 10%
                            fps = max(1, fps * 0.9)
                            frame_delay = int(1000000 / fps)
                        elif key == ord('f'):  # Reset to original FPS
                            fps = original_fps
                            frame_delay = int(1000000 / fps)

                    if mode == 'auto' and not is_paused:
                        frame_idx += 1
                        last_frame_time = t.time()
                    elif mode == 'manual' and key == 32:  # SPACE in manual mode
                        frame_idx += 1

                except Exception as e:
                    failed_frames += 1
                    logging.error(f"Error in frame {frame_idx + 1}: {str(e)}")
                    frame_idx += 1
                    continue

        # Print statistics
        logging.info("\nVideo processing completed:")
        logging.info(f"Total frames: {total_frames}")
        logging.info(f"Processed frames: {processed_frames}")
        logging.info(f"Failed frames: {failed_frames}")
        logging.info(f"Success rate: {(processed_frames / total_frames) * 100:.2f}%")

        cam.release()
        return True

def main():
    """
    Main execution function to process sign language videos.

    Workflow:
    1. Configure input/output paths
    2. Create VideoProcessor instance
    3. Get FPS settings from user
    4. Process each video and associated gloss
    5. Generate CSV output files
    6. Log processing statistics
    """
    # Configure paths
    input_videos_folder = "data/live/video"
    output_videos_folder = "data/live_data.csv"
    os.makedirs(output_videos_folder, exist_ok=True)

    # Create processor instance
    processor = VideoProcessor(accuracy=0.3)



    # Process each subfolder
    for subfolder_name in os.listdir(input_videos_folder):
        subfolder_path = os.path.join(input_videos_folder, subfolder_name)

        if not os.path.isdir(subfolder_path):
            continue

        # Find video and gloss files
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

        # Read gloss text
        with open(gloss_file, 'r', encoding='utf-8') as f:
            gloss_text = f.read().strip()

        # Process video with selected FPS
        csv_file_path = os.path.join(output_videos_folder, f"{subfolder_name}_traindata.csv")

        logging.info(f"Processing video: {subfolder_name}")
        if processor.process_video(video_file, csv_file_path, subfolder_name, gloss_text):
            logging.info(f"Successfully processed: {subfolder_name}")
        else:
            logging.error(f"Error processing: {subfolder_name}")


def record_video():
    """
    Records video and saves it for processing.

    Current Time: 2025-06-05 20:25:48
    User: CodeProTech
    """
    timestamp = "2025-06-05 20:25:48"
    user = "CodeProTech"

    print(f"\nAufnahme gestartet von {user}")
    print(f"Zeitstempel: {timestamp}\n")

    # Pfade definieren
    video_folder = "data/live/video"
    video_path = os.path.join(video_folder, "recorded_video.mp4")
    csv_path = "data/live/live_dataset.csv"

    # Verzeichnisse erstellen
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Video-Ordner leeren
    for filename in os.listdir(video_folder):
        file_path = os.path.join(video_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Fehler beim Löschen von {file_path}: {e}')

    # Kamera öffnen
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden")
        return None, 0

    # Video-Eigenschaften
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0

    # VideoWriter erstellen
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    print("Steuerung:")
    print("SPACE - Aufnahme starten/pausieren")
    print("Q     - Aufnahme beenden")

    recording = False
    frame_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Status-Overlay erstellen
        overlay = frame.copy()

        # Status-Box zeichnen
        cv2.rectangle(overlay, (0, 0), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Status-Informationen hinzufügen
        status_text = "AUFNAHME LÄUFT" if recording else "BEREIT - SPACE zum Starten"
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if recording else (0, 255, 0), 2)

        # Zusätzliche Informationen
        cv2.putText(frame, f"Frames: {frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"User: {user}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {timestamp}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Frame anzeigen
        cv2.imshow('Aufnahme', frame)

        # Tasteneingabe prüfen
        key = cv2.waitKey(1)

        if key == ord(' '):  # Leertaste
            recording = not recording
            if recording:
                print("Aufnahme läuft...")
            else:
                print("Aufnahme pausiert")

        if recording:
            out.write(frame)
            frame_count += 1

        if key == ord('q'):  # 'q' zum Beenden
            break

    # Aufräumen
    cam.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nAufnahme beendet:")
    print(f"Frames aufgenommen: {frame_count}")
    print(f"Video gespeichert: {video_path}")

    return video_path, frame_count


def process_video_to_csv(video_path, output_csv, frame_count):
    """
    Verarbeitet das aufgenommene Video und speichert die Keypoints im CSV.
    """
    # VideoProcessor erstellen
    processor = VideoProcessor(accuracy=0.3)

    print("\nStarte Keypoint-Analyse...")

    # Video verarbeiten
    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        print(f"Fehler: Video konnte nicht geöffnet werden: {video_path}")
        return False

    # CSV vorbereiten
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Headers nur für Frame und Keypoints
        headers = ["Frame"]
        headers += [f"pose_{i}_{axis}" for i in range(33) for axis in ['x', 'y']]
        headers += [f"hand_{i}_{axis}" for i in range(42) for axis in ['x', 'y']]
        headers += [f"face_{i}_{axis}" for i in range(468) for axis in ['x', 'y']]
        writer.writerow(headers)

        frame_idx = 0
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            # Fortschritt anzeigen
            progress = (frame_idx + 1) / frame_count * 100
            print(f"\rVerarbeite Frame {frame_idx + 1}/{frame_count} ({progress:.1f}%)", end="")

            # Frame verarbeiten
            current_data, processed_frame = processor.process_frame(frame, frame_idx)

            if current_data is not None:
                # Nur Frame-Nummer und Keypoints speichern
                writer.writerow(current_data)

                # Verarbeitetes Frame anzeigen
                cv2.imshow('Verarbeitung', processed_frame)

                if cv2.waitKey(1) == ord('q'):
                    break

            frame_idx += 1

    cam.release()
    cv2.destroyAllWindows()

    print(f"\n\nVerarbeitung abgeschlossen:")
    print(f"CSV gespeichert: {output_csv}")
    print(f"Verarbeitete Frames: {frame_idx}")

    return True


def main():
    """
    Hauptfunktion für Aufnahme und Verarbeitung.
    """
    try:
        # Video aufnehmen
        video_path, frame_count = record_video()

        if frame_count > 0:
            # Video verarbeiten und CSV erstellen
            csv_path = "data/live/live_dataset.csv"
            if process_video_to_csv(video_path, csv_path, frame_count):
                print("\nProgramm erfolgreich beendet.")
            else:
                print("\nFehler bei der Verarbeitung.")
        else:
            print("Keine Frames aufgenommen, Verarbeitung übersprungen.")

    except Exception as e:
        print(f"\nEin Fehler ist aufgetreten: {str(e)}")
    finally:
        cv2.destroyAllWindows()

