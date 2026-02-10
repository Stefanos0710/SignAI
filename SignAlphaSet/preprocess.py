import logging
import os
import time
import cv2
import numpy as np
import mediapipe as mp

# INFO: Test later with 2D keypoints -> better or not?: (x, y) vs (x, y, z)

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
    model_complexity=2
)

# folder of the alphabet-dataset
dataset_folder = "SignAlphaSet/data/SignAlphaSet/SignAlphaSet"

# extract keypoints from image
def extract_keypoints(image_path):
    # load the image
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Could not read image: {image_path}")
        return None

    # convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # process the image with MediaPipe Hands
    results = hands.process(image_rgb)

    # check if any hand is detected
    if not results.multi_hand_landmarks:
        logging.warning(f"No hand detected in image: {image_path}")
        return None
    
    # get first detected hands landmarks
    hand = results.multi_hand_landmarks[0]

    keypoints = []
    for landmark in hand.landmark:
        keypoints.append((landmark.x, landmark.y, landmark.z))

    keypoints = np.array(keypoints)
    return keypoints

# center the keypoints
def center_keypoints(keypoints):
    pass

# normalize the keypoints
def normalize_keypoints(keypoints):
    pass

# save the keypoints to a file
def save_keypoints(keypoints, save_path):
    pass

# split the dataset into train, val and test
def split_dataset(dataset_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    pass

if __name__ == "__main__":
    alphabet = sorted(os.listdir(dataset_folder))  #  sort to["A", "B", ..., "Z"]

    for class_name in alphabet:
        class_path = os.path.join(dataset_folder, class_name)
        if os.path.isdir(class_path):
            files = os.listdir(class_path)
            logging.info(f"Class '{class_name}' has {len(files)} files.")

            # print all files from class_name
            for file_name in files:
                # extract keypoints from current image
                tmp_keypoints = extract_keypoints(os.path.join(class_path, file_name))
                print(os.path.join(class_path, file_name))

                # center the keypoints
                tmp_keypoints = center_keypoints(tmp_keypoints)

                # normalize the keypoints
                tmp_keypoints = normalize_keypoints(tmp_keypoints)

                # save the keypoints to a file
                save_path = os.path.join("SignAlphaSet/data/tmp_keypoints", class_name)
                save_keypoints(tmp_keypoints, save_path)
                
    # split the dataset into train, val and test
    split_dataset(dataset_folder)
