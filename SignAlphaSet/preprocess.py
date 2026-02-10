import os
# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import mediapipe as mp
import logging
import time

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ----------------------------
# Initialization of MediaPipe Hands
# ----------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    model_complexity=1
)

# ----------------------------
# Folder of the alphabet-dataset
# ----------------------------
dataset_folder = "SignAlphaSet/data/SignAlphaSet/SignAlphaSet"

# ----------------------------
# Lists to store keypoints and labels
# ----------------------------
all_keypoints = []
all_labels = []
failed_files = []

# ----------------------------
# Extract keypoints from image
# ----------------------------
def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Could not read image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        logging.warning(f"No hand detected in image: {image_path}")
        return None

    hand = results.multi_hand_landmarks[0]
    keypoints = np.array([(lm.x, lm.y, lm.z) for lm in hand.landmark], dtype=np.float32)
    return keypoints

# ----------------------------
# Center the keypoints
# ----------------------------
def center_keypoints(keypoints):
    # get wrist keypoint (landmark 0) => (0,0,0)
    wrist_keypoint = keypoints[0]
    centered_keypoints = keypoints - wrist_keypoint
    return centered_keypoints

# ----------------------------
# Normalize the keypoints
# ----------------------------
def normalize_keypoints(keypoints):
    wrist_keypoint = keypoints[0]
    middle_finger_tip = keypoints[12]
    scale = np.linalg.norm(middle_finger_tip - wrist_keypoint)
    if scale < 1e-8:
        logging.warning("Distance too small, skipping normalization.")
        return keypoints
    normalized_keypoints = keypoints / scale
    return normalized_keypoints

# ----------------------------
# Split dataset into train, val, test and save as npz files
# ----------------------------
def create_dataset(X, y, output_folder="SignAlphaSet/data/processed_dataset", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # shuffle first
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    N = len(X)
    train_end = int(train_ratio * N)
    val_end = int((train_ratio + val_ratio) * N)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # save as npz in output folder
    np.savez_compressed(os.path.join(output_folder, "train_data.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(output_folder, "val_data.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(output_folder, "test_data.npz"), X=X_test, y=y_test)

    logging.info(f"Datasets saved in folder: {output_folder}")
    logging.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")


# ----------------------------
# Main processing
# ----------------------------
if __name__ == "__main__":
    alphabet = sorted(os.listdir(dataset_folder))  # ["A","B",...,"Z"]
    class_to_idx = {c: i for i, c in enumerate(alphabet)}

    total = 0

    for class_name in alphabet:
        class_path = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        files = os.listdir(class_path)
        logging.info(f"Class '{class_name}' has {len(files)} files.")

        for file_name in files:
            file_path = os.path.join(class_path, file_name)

            # extract keypoints
            tmp_keypoints = extract_keypoints(file_path)

            # check if keypoints are fine
            if tmp_keypoints is None:
                logging.warning(f"Skipping file '{file_name}' in class '{class_name}' due to keypoint extraction failure.")
                failed_files.append(file_path)
                continue

            # center and normalize
            tmp_keypoints = center_keypoints(tmp_keypoints)
            tmp_keypoints = normalize_keypoints(tmp_keypoints)

            # append to dataset
            all_keypoints.append(tmp_keypoints)
            all_labels.append(class_to_idx[class_name])
            total += 1

            # processed count logging/progress
            if total % 50 == 0:
                logging.info(f"Processed {total} samples...")

    # =================================================
    # Convert to arrays
    # =================================================
    X = np.array(all_keypoints, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    logging.info(f"Valid samples: {len(X)}")
    logging.info(f"Failed samples: {len(failed_files)}")

    # =================================================
    # Split and save train/val/test datasets
    # =================================================
    create_dataset(X, y)
