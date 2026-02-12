import os
import json
import base64
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from flask import Flask, logging, render_template, request, jsonify
import logging

# Create the Flask application
app = Flask(__name__)

# paths to the model and label files
model_path = "SignAlphaSet\models\signalphaset_v1.keras"
labels_path = "SignAlphaSet\models\signalphaset_label_map_v1.json"

# try to load model
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# try to load labels
idx_to_label = {}
try:
    with open(labels_path, 'r') as f:
        label_map = json.load(f)
        # invert the label map to get index to label mapping
        idx_to_label = {v: k for k, v in label_map.items()}
    print(f"Labels loaded: {len(idx_to_label)} classes")
except Exception as e:
    print(f"Error loading labels: {e}")

# ----------------------------
# Initialization of MediaPipe Hands (from process.py)
# ----------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    model_complexity=1
)

# ----------------------------
# Extract keypoints from image (from process.py)
# ----------------------------
def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Could not read image: {image_path}")
        return None, "read_failed"

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        logging.warning(f"No hand detected in image: {image_path}")
        return None, "no_hand"

    hand = results.multi_hand_landmarks[0]
    keypoints = np.array([(lm.x, lm.y, lm.z) for lm in hand.landmark], dtype=np.float32)
    return keypoints, None

# ----------------------------
# Center the keypoints (from process.py)
# ----------------------------
def center_keypoints(keypoints):
    # get wrist keypoint (landmark 0) => (0,0,0)
    wrist_keypoint = keypoints[0]
    centered_keypoints = keypoints - wrist_keypoint
    return centered_keypoints

# ----------------------------
# Normalize the keypoints (from process.py)
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

def preprocess_keypoints(image_path):
    # 1. Extract keypoints
    keypoints = extract_keypoints(image_path)

    # 2. Center keypoints
    centered_keypoints = center_keypoints(keypoints)

    # 3. Normalize keypoints
    normalized_keypoints = normalize_keypoints(centered_keypoints)

    # 4. Reshape for model input
    input_data = np.expand_dims(normalized_keypoints, axis=0)

    return normalized_keypoints

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
