import csv
import logging
import multiprocessing
import os

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# MediaPipe solution references (instances are created per worker process)
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# Placeholders — initialized inside each worker via init_worker()
hands = None
face  = None
pose  = None

# Number of parallel workers
NUM_WORKERS = 12

FACE_LANDMARKS = [
    # ===== EYEBROWS =====
    46,53,52,65,55,70,63,105,66,107,          # left eyebrow
    276,283,282,295,285,336,296,334,293,300,  # right eyebrow

    # ===== EYES =====
    33,160,158,133,153,144,163,7,246,161,159,157,      # left eye
    362,385,387,263,373,380,390,249,466,388,386,384,   # right eye

    # ===== NOSE =====
    1,2,98,327,168,

    # ===== MOUTH / LIPS  =====
    61,146,91,181,84,17,314,405,321,375,291,308,  # outer lip contour
    78,95,88,178,87,14,317,402,318,324,           # inner lip contour
    185,40,39,37,0,267,269,270,409,               # upper lip region
    191,80,81,82,13,312,311,310,415,              # lower lip region

    # ===== CHEEKS =====
    50,280,187,425
]


POSE_LANDMARKS = [
    0,   # nose
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16   # right wrist
]


def init_worker():
    """Initialize one set of MediaPipe objects per worker process."""
    global hands, face, pose
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
    )
    face = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def process_subfolder(subfolder_path):
    """Process a single subfolder: extract keypoints and save to CSV."""
    subfolder_name = os.path.basename(subfolder_path)
    logging.info(f"Currently processing folder: {subfolder_name}")

    gloss = extract_gloss(subfolder_path)
    logging.info(f"  -> Extracted gloss: {gloss}")

    video_file = None
    for file_name in os.listdir(subfolder_path):
        if file_name.lower().endswith('.mp4'):
            video_file = os.path.join(subfolder_path, file_name)
            break

    if video_file is None:
        logging.warning(f"  -> No .mp4 file found in {subfolder_path}, skipping.")
        return

    frames = extract_frames(video_file)
    if not frames:
        logging.warning(f"  -> No frames extracted, skipping.")
        return

    pose_keypoints = extract_pose_keypoints(frames)
    hand_keypoints = extract_hand_keypoints(frames)
    face_keypoints = extract_face_keypoints(frames)
    avg_left_shoulder, avg_right_shoulder = average_shoulder_kp(video_file)

    save_in_csv(
        pose_keypoints,
        hand_keypoints,
        face_keypoints,
        gloss,
        subfolder_name,
        avg_left_shoulder,
        avg_right_shoulder,
    )


def extract_gloss(folder_path):
    # look for .txt file in the folder and extract the gloss
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                gloss = f.read().strip()
                return gloss
    logging.warning(f"  -> No gloss text file found in {folder_path}")
    return "not_found"


def extract_frames(video_path):
    """Read a video file and return a list of all frames."""
    frames = []
    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        logging.error(f"  -> Could not open video: {video_path}")
        return frames
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frames.append(frame)
    cam.release()
    logging.info(f"  -> Extracted {len(frames)} frames from video")
    return frames


def extract_pose_keypoints(frames):
    """Extract pose keypoints for each frame, filtered to POSE_LANDMARKS indices."""
    all_pose_keypoints = []
    for idx, frame in enumerate(frames):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True

        if results.pose_landmarks:
            all_lm = results.pose_landmarks.landmark
            keypoints = [(all_lm[i].x, all_lm[i].y, all_lm[i].z)
                         for i in POSE_LANDMARKS]
        else:
            keypoints = None
        all_pose_keypoints.append((idx, keypoints))
    return all_pose_keypoints


def extract_hand_keypoints(frames):
    """Extract hand keypoints for each frame, separated into left and right hand.
    Returns list of (idx, left_hand_kp, right_hand_kp) per frame.
    Each hand is a list of 21 (x, y, z) tuples, or None if not detected.
    """
    EMPTY_HAND = [(0.0, 0.0, 0.0)] * 21
    all_hand_keypoints = []
    for idx, frame in enumerate(frames):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        left_hand = None
        right_hand = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                if label == 'Left':
                    left_hand = keypoints
                else:
                    right_hand = keypoints

        all_hand_keypoints.append((idx, left_hand, right_hand))
    return all_hand_keypoints


def extract_face_keypoints(frames):
    """Extract face mesh keypoints for each frame, filtered to FACE_LANDMARKS indices."""
    all_face_keypoints = []
    for idx, frame in enumerate(frames):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face.process(image)
        image.flags.writeable = True

        if results.multi_face_landmarks:
            all_lm = results.multi_face_landmarks[0].landmark
            keypoints = [(all_lm[i].x, all_lm[i].y, all_lm[i].z)
                         for i in FACE_LANDMARKS]
        else:
            keypoints = None
        all_face_keypoints.append((idx, keypoints))
    return all_face_keypoints


def center_keypoints(keypoints, avg_left_shoulder=None, avg_right_shoulder=None):
    """Center all keypoints around the midpoint between the shoulders in all 3 dimensions (x, y, z).
    Expects keypoints as a numpy array of shape (N, 3) with pose landmarks first,
    where index 1 = left shoulder and index 2 = right shoulder (per POSE_LANDMARKS).
    If average shoulder positions are provided, use them (video-wise centering).
    """
    kp = np.array(keypoints, dtype=float)
    left_shoulder = np.array(avg_left_shoulder, dtype=float) if avg_left_shoulder is not None else kp[1]
    right_shoulder = np.array(avg_right_shoulder, dtype=float) if avg_right_shoulder is not None else kp[2]
    center = (left_shoulder + right_shoulder) / 2.0
    centered_keypoints = kp - center
    return centered_keypoints


def normalize_keypoints(keypoints, avg_left_shoulder=None, avg_right_shoulder=None):
    """Normalize keypoints so that the distance between the shoulders equals 1.
    Expects keypoints as a numpy array of shape (N, 3) with pose landmarks first,
    where index 1 = left shoulder and index 2 = right shoulder (per POSE_LANDMARKS).
    If average shoulder positions are provided, use them (video-wise normalization scale).
    """
    kp = np.array(keypoints, dtype=float)
    left_shoulder = np.array(avg_left_shoulder, dtype=float) if avg_left_shoulder is not None else kp[1]
    right_shoulder = np.array(avg_right_shoulder, dtype=float) if avg_right_shoulder is not None else kp[2]
    dist = np.linalg.norm(right_shoulder - left_shoulder)
    if dist == 0:
        return kp
    normalized_keypoints = kp / dist
    return normalized_keypoints


def apply_temporal_savgol_smoothing(sequence, window_length=9, polyorder=2):
    """Apply temporal Savitzky-Golay smoothing over frames.

    sequence shape: (num_frames, num_landmarks, 3)
    Smoothing is applied along the frame axis to reduce MediaPipe jitter
    while preserving movement dynamics better than a simple moving average.
    """
    seq = np.array(sequence, dtype=float)
    num_frames = seq.shape[0]

    # Need at least polyorder + 2 points; keep an odd window for SavGol.
    if num_frames < (polyorder + 2):
        return seq

    max_odd_window = num_frames if num_frames % 2 == 1 else num_frames - 1
    if max_odd_window < (polyorder + 2):
        return seq

    window = min(window_length, max_odd_window)
    if window % 2 == 0:
        window -= 1

    if window <= polyorder:
        window = polyorder + 1
        if window % 2 == 0:
            window += 1
    if window > max_odd_window:
        return seq

    # Apply filter for all landmarks and xyz channels over time.
    smoothed = savgol_filter(seq, window_length=window, polyorder=polyorder, axis=0, mode="interp")
    return smoothed


def save_in_csv(
    pose_keypoints,
    hand_keypoints,
    face_keypoints,
    gloss,
    subfolder_name,
    avg_left_shoulder=None,
    avg_right_shoulder=None,
):
    """Save the extracted keypoints into a CSV file.
    Format: name, GLOSS, Frame, pose keypoints, left hand keypoints, right hand keypoints, face keypoints
    """
    output_folder = os.path.join("data", "train_data")
    os.makedirs(output_folder, exist_ok=True)
    csv_file_path = os.path.join(output_folder, f"{subfolder_name}_traindata.csv")

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Build header using actual landmark indices from the filter lists
        header = ["name", "GLOSS", "Frame"]
        header += [f"pose_{i}_{a}" for i in POSE_LANDMARKS for a in ["x", "y", "z"]]
        header += [f"left_hand_{i}_{a}" for i in range(21) for a in ["x", "y", "z"]]
        header += [f"right_hand_{i}_{a}" for i in range(21) for a in ["x", "y", "z"]]
        header += [f"face_{i}_{a}" for i in FACE_LANDMARKS for a in ["x", "y", "z"]]
        writer.writerow(header)

        n_pose = len(POSE_LANDMARKS)
        n_hand = 21
        n_face = len(FACE_LANDMARKS)

        frame_ids = []
        all_frames_keypoints = []

        for i in range(len(pose_keypoints)):
            frame_idx = pose_keypoints[i][0]

            # Build per-part arrays (N, 3)
            pose_kp = pose_keypoints[i][1]
            pose_arr = np.array(pose_kp, dtype=float) if pose_kp else np.zeros((n_pose, 3))

            _, left_hand, right_hand = hand_keypoints[i]
            left_arr  = np.array(left_hand,  dtype=float) if left_hand  else np.zeros((n_hand, 3))
            right_arr = np.array(right_hand, dtype=float) if right_hand else np.zeros((n_hand, 3))

            face_kp = face_keypoints[i][1]
            face_arr = np.array(face_kp, dtype=float) if face_kp else np.zeros((n_face, 3))

            # Combine all landmarks, then center and normalize (only when pose is detected)
            all_kp = np.concatenate([pose_arr, left_arr, right_arr, face_arr], axis=0)
            if pose_kp:
                all_kp = center_keypoints(all_kp, avg_left_shoulder, avg_right_shoulder)
                all_kp = normalize_keypoints(all_kp, avg_left_shoulder, avg_right_shoulder)

            frame_ids.append(frame_idx)
            all_frames_keypoints.append(all_kp)

        # Fill missing landmarks first, then smooth to reduce temporal jitter.
        interpolated_sequence = interpolate_missing_keypoints(all_frames_keypoints)
        smoothed_sequence = apply_temporal_savgol_smoothing(interpolated_sequence, window_length=9, polyorder=2)

        for i, frame_idx in enumerate(frame_ids):
            all_kp = smoothed_sequence[i]

            # Split back and flatten
            flat_pose = all_kp[:n_pose].flatten().tolist()
            flat_left = all_kp[n_pose:n_pose + n_hand].flatten().tolist()
            flat_right = all_kp[n_pose + n_hand:n_pose + 2 * n_hand].flatten().tolist()
            flat_face = all_kp[n_pose + 2 * n_hand:].flatten().tolist()

            writer.writerow([subfolder_name, gloss, frame_idx]
                            + flat_pose + flat_left + flat_right + flat_face)

    logging.info(f"  -> Saved keypoints to {csv_file_path}")

def average_shoulder_kp(video_file):
    """Calculate the average shoulder keypoint position from each video.
    This is used for centering and normalizing the keypoints videowise NOT framewise.
    """

    # get all frames from the video
    extracted_frames = extract_frames(video_file)
    if not extracted_frames:
        return None, None

    extracted_pose_kp = extract_pose_keypoints(extracted_frames)

    left_shoulders = []
    right_shoulders = []

    for _, pose_kp in extracted_pose_kp:
        if not pose_kp:
            continue
        left_shoulders.append(np.array(pose_kp[1], dtype=float))   # index 1 in POSE_LANDMARKS
        right_shoulders.append(np.array(pose_kp[2], dtype=float))  # index 2 in POSE_LANDMARKS

    if not left_shoulders or not right_shoulders:
        return None, None

    avg_left_shoulder = np.mean(left_shoulders, axis=0)
    avg_right_shoulder = np.mean(right_shoulders, axis=0)

    return avg_left_shoulder, avg_right_shoulder

def interpolate_missing_keypoints(sequence):
    """Interpolate missing keypoints (0,0,0) in the sequence using linear interpolation.
    """
    seq = np.array(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[0] == 0:
        return seq

    num_frames, num_landmarks, num_dims = seq.shape
    frame_idx = np.arange(num_frames)

    # A landmark is treated as missing in a frame when all xyz values are exactly zero.
    missing_mask = np.all(seq == 0.0, axis=2)

    for lm_idx in range(num_landmarks):
        missing_frames = missing_mask[:, lm_idx]
        if not np.any(missing_frames):
            continue

        valid_frames = ~missing_frames
        if not np.any(valid_frames):
            # No valid observation for this landmark in the full clip.
            continue

        valid_x = frame_idx[valid_frames]
        for dim_idx in range(num_dims):
            valid_y = seq[valid_frames, lm_idx, dim_idx]
            seq[:, lm_idx, dim_idx] = np.interp(frame_idx, valid_x, valid_y)

    return seq


def maybe_clean_output_folder():
    """Optionally delete old output CSV files."""
    output_folder = os.path.join("data", "train_data")
    os.makedirs(output_folder, exist_ok=True)

    try:
        answer = input("Delete old output CSVs? (y/n): ").strip().lower()
    except EOFError:
        # Non-interactive run: keep existing files by default.
        logging.info("No interactive input available; keeping existing output files.")
        return

    if answer not in {"y", "yes"}:
        logging.info("Keeping existing output files.")
        return

    removed = 0
    
    for file_name in os.listdir(output_folder):
        if file_name.lower().endswith(".csv"):
            file_path = os.path.join(output_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                removed += 1

    logging.info(f"Removed {removed} existing CSV file(s) from {output_folder}.")


if __name__ == "__main__":
    logging.info("this script is made for the preprocessing of datasets (Phoenix 2014 T)")
    maybe_clean_output_folder()

    raw_vid = os.path.join("data", "raw_data", "train")

    subfolders = [
        os.path.join(raw_vid, d)
        for d in sorted(os.listdir(raw_vid))
        if os.path.isdir(os.path.join(raw_vid, d))
    ]

    logging.info(f"Found {len(subfolders)} subfolders — starting {NUM_WORKERS} workers...")

    with multiprocessing.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        pool.map(process_subfolder, subfolders)
