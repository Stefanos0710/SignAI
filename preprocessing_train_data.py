import cv2
import mediapipe as mp
import csv
import os
import logging
import numpy as np


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# set up md settings (hand and pose tracking, face)
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# set up hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5, # upgrade this later with tests
    model_complexity=0, # upgrade this later with tests
)

#set up face mesh 
face = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.3, # upgrade this later with tests
    min_tracking_confidence=0.3, # upgrade this later with tests

)

# set up the 
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0, # upgrade this later with tests
    min_detection_confidence=0.5, # upgrade this later with tests
    min_tracking_confidence=0.5 # upgrade this later with tests
)

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

def save_in_csv(pose_keypoints, hand_keypoints, face_keypoints, gloss, subfolder_name):
    """Save the extracted keypoints into a CSV file.
    Format: name, GLOSS, Frame, pose keypoints, left hand keypoints, right hand keypoints, face keypoints
    """
    output_folder = os.path.join("data", "train_data")
    os.makedirs(output_folder, exist_ok=True)
    csv_file_path = os.path.join(output_folder, f"{subfolder_name}_traindata.csv")

    EMPTY_HAND = [0.0] * (21 * 3)                    # 21 landmarks * (x, y, z)
    EMPTY_POSE = [0.0] * (len(POSE_LANDMARKS) * 3)   # filtered landmarks * (x, y, z)
    EMPTY_FACE = [0.0] * (len(FACE_LANDMARKS) * 3)   # filtered landmarks * (x, y, z)

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Build header using actual landmark indices from the filter lists
        header = ["name", "GLOSS", "Frame"]
        header += [f"pose_{i}_{a}" for i in POSE_LANDMARKS for a in ["x", "y", "z"]]
        header += [f"left_hand_{i}_{a}" for i in range(21) for a in ["x", "y", "z"]]
        header += [f"right_hand_{i}_{a}" for i in range(21) for a in ["x", "y", "z"]]
        header += [f"face_{i}_{a}" for i in FACE_LANDMARKS for a in ["x", "y", "z"]]
        writer.writerow(header)

        for i in range(len(pose_keypoints)):
            frame_idx = pose_keypoints[i][0]

            # Pose: flatten (x, y, z) for each landmark
            pose_kp = pose_keypoints[i][1]
            if pose_kp:
                flat_pose = [v for lm in pose_kp for v in lm]
            else:
                flat_pose = EMPTY_POSE

            # Hands: left and right separately
            _, left_hand, right_hand = hand_keypoints[i]
            flat_left  = [v for lm in left_hand  for v in lm] if left_hand  else EMPTY_HAND
            flat_right = [v for lm in right_hand for v in lm] if right_hand else EMPTY_HAND

            # Face: flatten (x, y, z) for each landmark
            face_kp = face_keypoints[i][1]
            if face_kp:
                flat_face = [v for lm in face_kp for v in lm]
            else:
                flat_face = EMPTY_FACE

            writer.writerow([subfolder_name, gloss, frame_idx]
                            + flat_pose + flat_left + flat_right + flat_face)

    logging.info(f"  -> Saved keypoints to {csv_file_path}")

if __name__ == "__main__":
    logging.info("this script is made for the preprocessing of datasets (Phoenix 2014 T)")

    raw_vid = os.path.join("data", "raw_data", "train")

    # go through all the files in the raw data folder and process them one by one
    for supfolder in sorted(os.listdir(raw_vid)):
        subfolder_name = supfolder
        subfolder_path = os.path.join(raw_vid, subfolder_name)

        if not os.path.isdir(subfolder_path):
            continue

        logging.info(f"Currently processing folder: {supfolder}")

        # get gloss
        gloss = extract_gloss(subfolder_path)
        logging.info(f"  -> Extracted gloss: {gloss}")

        # find video file in the subfolder
        video_file = None
        for file_name in os.listdir(subfolder_path):
            if file_name.lower().endswith('.mp4'):
                video_file = os.path.join(subfolder_path, file_name)
                break

        if video_file is None:
            logging.warning(f"  -> No .mp4 file found in {subfolder_path}, skipping.")
            continue

        # extract all frames from the video
        frames = extract_frames(video_file)
        if not frames:
            logging.warning(f"  -> No frames extracted, skipping.")
            continue

        # extract keypoints from the frames
        pose_keypoints = extract_pose_keypoints(frames)
        hand_keypoints = extract_hand_keypoints(frames)
        face_keypoints = extract_face_keypoints(frames)

        # save keypoints to csv
        save_in_csv(pose_keypoints, hand_keypoints, face_keypoints, gloss, subfolder_name)
