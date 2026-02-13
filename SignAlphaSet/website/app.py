import os
import json
import base64
import numpy as np
import cv2

# Force CPU for TensorFlow to avoid conflicts/hangs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import mediapipe as mp
from flask import Flask, logging, render_template, request, jsonify, send_from_directory
import logging

# Create the Flask application
app = Flask(__name__)

# -------------------------------------------------------------
# 1. ROBUST PATH SETUP
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'signalphaset_v1.keras')

# Path to the raw dataset to get class names
DATASET_DIR = os.path.join(BASE_DIR, '..', 'data', 'SignAlphaSet', 'SignAlphaSet') 

# path to the dictionary images
PICTURES_DIR = os.path.join(BASE_DIR, 'pic')

logging.basicConfig(level=logging.INFO)
logging.info("="*50)
logging.info(f"Starting App")
logging.info(f"Model Path: {MODEL_PATH}")

# -------------------------------------------------------------
# 2. LOAD RESOURCES & LABELS
# -------------------------------------------------------------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("Model loaded.")
    except Exception as e:
        logging.error(f"Could not load model: {e}")
else:
    logging.critical(f"Model file NOT found at {MODEL_PATH}")

# load and map class labels from dataset folder
idx_to_label = {}
try:
    if os.path.exists(DATASET_DIR):
        # The model was trained on sorted folder names
        classes = sorted(os.listdir(DATASET_DIR))
        # create map: 0 -> "A", 1 -> "B", etc.
        idx_to_label = {i: name for i, name in enumerate(classes)}
        logging.info(f"Loaded {len(idx_to_label)} labels from dataset folder.")
        logging.debug(f"First few labels: {list(idx_to_label.values())[:5]}")

    else:
        logging.warning(f"Dataset folder not found at {DATASET_DIR}")
        import string
        letters = list(string.ascii_uppercase)
        idx_to_label = {i: letter for i, letter in enumerate(letters)}
        logging.debug("Using fallback A-Z labels.")

except Exception as e:
    logging.error(f"Could not load labels: {e}")

logging.info("="*50)

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
def extract_keypoints(image):
    if image is None:
        return None, "read_failed", {}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None, "no_hand", {}

    hand = results.multi_hand_landmarks[0]
    keypoints = np.array([(lm.x, lm.y, lm.z) for lm in hand.landmark], dtype=np.float32)
    
    # Get Handedness info
    handedness_info = {}
    if results.multi_handedness:
        h_class = results.multi_handedness[0].classification[0]
        handedness_info = {
            'label': h_class.label,
            'score': float(h_class.score)
        }

    return keypoints, None, handedness_info

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
        return keypoints, 1.0
    normalized_keypoints = keypoints / scale
    return normalized_keypoints, scale

def preprocess_keypoints(image, debug=False):
    # 1. Extract keypoints
    if debug:
        pass

    keypoints, error, hand_info = extract_keypoints(image)

    if error:
        return None, None, None, {}

    # 2. Center keypoints
    centered_keypoints = center_keypoints(keypoints)

    # 3. Normalize keypoints
    normalized_keypoints, scale = normalize_keypoints(centered_keypoints)

    # 4. Reshape for model input
    input_data = np.expand_dims(normalized_keypoints, axis=0)
    
    extra_info = {
        'handedness': hand_info,
        'scale_factor': float(scale),
        'input_shape': input_data.shape
    }

    # If debug is on, return intermediate data too
    if debug:
        return input_data, keypoints, normalized_keypoints, extra_info
        
    return input_data, None, None, extra_info

# Define a route for the home page
@app.route('/')
def home():
    labels = list(idx_to_label.values())
    return render_template('index.html', labels=labels)

@app.route('/dataset_img/<label>')
def dataset_img(label):
    # Check for png first, then jpg
    for ext in ['.png', '.jpg']:
        filename = f"{label}{ext}"
        if os.path.exists(os.path.join(PICTURES_DIR, filename)):
             return send_from_directory(PICTURES_DIR, filename)
    return "", 404

import time

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    if model is None:
        return jsonify({'error': 'Model not loaded', 'prediction': 'Error'}), 500

    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image received'}), 400

    debug_mode = data.get('debug', False)

    try:
        # 1. Decode image from Base64
        decode_start = time.time()
        image_data = data['image'].split(',')[1] 
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        decode_time = (time.time() - decode_start) * 1000

        # 2. Preprocessing
        prep_start = time.time()
        input_data, raw_kps, norm_kps, extra_info = preprocess_keypoints(image, debug=debug_mode)
        prep_time = (time.time() - prep_start) * 1000

        if input_data is None:
            return jsonify({'prediction': 'No Hand', 'confidence': 0.0})

        # 3. Prediction
        inference_start = time.time()
        prediction = model.predict(input_data, verbose=0)[0]
        inference_time = (time.time() - inference_start) * 1000
        
        predicted_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Get label from our fixed map
        predicted_label = idx_to_label.get(predicted_idx, f"Class {predicted_idx}")

        response = {
            'prediction': predicted_label,
            'confidence': confidence
        }

        if debug_mode:
            # Get Top 5 Predictions
            # argsort returns indices that sort the array, we take last 5 and reverse
            top_5_indices = prediction.argsort()[-5:][::-1]
            top_5 = []
            for idx in top_5_indices:
                top_5.append({
                    'label': idx_to_label.get(idx, str(idx)),
                    'confidence': float(prediction[idx])
                })
            
            response['top_5'] = top_5
            
            # Format numbers for cleanliness
            h_label = extra_info['handedness'].get('label', 'Unknown')
            h_score = extra_info['handedness'].get('score', 0.0)
            
            response['meta'] = {
                'handedness': f"{h_label} ({h_score*100:.1f}%)",
                'scale': f"{extra_info['scale_factor']:.4f}",
                'input_shape': str(extra_info['input_shape'])
            }

            response['timing'] = {
                'decode': f"{decode_time:.1f}ms",
                'preprocess': f"{prep_time:.1f}ms",
                'inference': f"{inference_time:.1f}ms",
                'total': f"{(time.time() - start_time) * 1000:.1f}ms"
            }

        if debug_mode and raw_kps is not None:
            # Create the hand cutout
            # Use lower quality for speed if needed, but cutout is small anyway
            h, w, _ = image.shape
            x_min, y_min = np.min(raw_kps[:, :2], axis=0)
            x_max, y_max = np.max(raw_kps[:, :2], axis=0)
            
            # Convert to pixels and add padding
            pad = 20
            x1 = max(0, int(x_min * w) - pad)
            y1 = max(0, int(y_min * h) - pad)
            x2 = min(w, int(x_max * w) + pad)
            y2 = min(h, int(y_max * h) + pad)
            
            cutout = image[y1:y2, x1:x2]
            
            # Encode cutout to base64
            # Use very low quality for speed (30)
            _, buffer = cv2.imencode('.jpg', cutout, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            cutout_b64 = base64.b64encode(buffer).decode('utf-8')
            
            response['debug_info'] = {
                'raw_landmarks': raw_kps.tolist(),
               # 'norm_landmarks': norm_kps.tolist(), # Skip sending this to save bandwidth
                'hand_cutout': f"data:image/jpeg;base64,{cutout_b64}"
            }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
