import os
import json
import base64
import re
import time
import threading
import string
import numpy as np
import cv2


"""
TODOs:
- show keypoints as reference, when clicked on the example panals

"""



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
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
CAPTURED_DIR = os.path.join(BASE_DIR, 'pic', 'captured')

# Path to the raw dataset to get class names
DATASET_DIR = os.path.join(BASE_DIR, '..', 'data', 'SignAlphaSet', 'SignAlphaSet') 

# path to the dictionary images
PICTURES_DIR = os.path.join(BASE_DIR, 'pic')

logging.basicConfig(level=logging.INFO)
logging.info("="*50)
logging.info(f"Starting App")
logging.info(f"Models Dir: {MODELS_DIR}")

NUM_LANDMARKS = 21
COORD_DIMS = 3
BASE_KEYPOINT_FEATURES = NUM_LANDMARKS * COORD_DIMS
REC_LETTERS = list(string.ascii_uppercase)

EXTRA_FEATURE_SPECS = [
    ("index_middle_tip", 8, 12),
    ("middle_ring_tip", 12, 16),
    ("thumb_index_dip", 4, 7),
    ("thumb_index_pip", 4, 6),
    ("thumb_middle_dip", 4, 11),
    ("thumb_middle_pip", 4, 10),
    ("thumb_ring_dip", 4, 15),
    ("thumb_ring_pip", 4, 14),
    ("thumb_pinky_dip", 4, 19),
    ("thumb_pinky_pip", 4, 18),
]

rec_state_lock = threading.Lock()
rec_session = None


def _status_text_for_phase(phase):
    mapping = {
        "waiting": "Warte",
        "countdown": "Warte",
        "capturing": "Aufnahme läuft",
        "pause": "Pause",
        "done": "Fertig",
        "stopped": "Gestoppt",
        "error": "Fehler",
    }
    return mapping.get(phase, "Warte")


def _build_rec_state_payload(session):
    now = time.time()
    seconds_remaining = 0.0
    if session.get("next_phase_at"):
        seconds_remaining = max(0.0, float(session["next_phase_at"]) - now)

    current_letter = None
    if 0 <= session["current_letter_index"] < len(session["letters"]):
        current_letter = session["letters"][session["current_letter_index"]]

    return {
        "phase": session["phase"],
        "status_text": _status_text_for_phase(session["phase"]),
        "current_letter": current_letter,
        "current_letter_index": session["current_letter_index"],
        "letters_total": len(session["letters"]),
        "current_progress": session["accepted_count_current"],
        "target_count": session["target_count"],
        "seconds_remaining": round(seconds_remaining, 2),
        "fps": session["fps"],
        "similarity_threshold": session["similarity_threshold"],
        "summary": session["summary"],
        "total_saved": session["total_saved"],
        "zip_available": False,
        "dataset_path": CAPTURED_DIR,
    }


def _ensure_capture_dirs():
    os.makedirs(CAPTURED_DIR, exist_ok=True)
    for letter in REC_LETTERS:
        os.makedirs(os.path.join(CAPTURED_DIR, letter), exist_ok=True)


def _list_existing_image_paths(letter):
    letter_dir = os.path.join(CAPTURED_DIR, letter)
    if not os.path.isdir(letter_dir):
        return []

    paths = []
    for file_name in os.listdir(letter_dir):
        lower = file_name.lower()
        if not (lower.endswith('.png') or lower.endswith('.jpg') or lower.endswith('.jpeg')):
            continue
        if not file_name.startswith(f"{letter}_"):
            continue
        paths.append(os.path.join(letter_dir, file_name))

    paths.sort()
    return paths


def _load_existing_vectors_for_letter(letter):
    vectors = []
    for image_path in _list_existing_image_paths(letter):
        image = cv2.imread(image_path)
        if image is None:
            continue

        keypoints, error, _ = extract_keypoints(image)
        if error:
            continue

        centered_keypoints = center_keypoints(keypoints)
        normalized_keypoints, _ = normalize_keypoints(centered_keypoints)
        vectors.append(normalized_keypoints.reshape(-1).astype(np.float32))

    return vectors


def _find_first_incomplete_index(summary, target_count):
    for index, letter in enumerate(REC_LETTERS):
        if int(summary.get(letter, 0)) < int(target_count):
            return index
    return len(REC_LETTERS)


def _activate_current_letter(session):
    if session["current_letter_index"] >= len(session["letters"]):
        session["phase"] = "done"
        session["next_phase_at"] = None
        session["accepted_vectors_current"] = []
        session["accepted_count_current"] = 0
        return

    current_letter = session["letters"][session["current_letter_index"]]
    existing_count = int(session["summary"].get(current_letter, 0))

    session["accepted_count_current"] = existing_count

    if existing_count >= session["target_count"]:
        next_index = _find_first_incomplete_index(session["summary"], session["target_count"])
        session["current_letter_index"] = next_index
        _activate_current_letter(session)
        return

    session["accepted_vectors_current"] = _load_existing_vectors_for_letter(current_letter)
    session["phase"] = "countdown"
    session["next_phase_at"] = time.time() + session["countdown_seconds"]


def _new_rec_session(fps, similarity_threshold, target_count):
    _ensure_capture_dirs()
    existing_summary = {}
    for letter in REC_LETTERS:
        existing_summary[letter] = len(_list_existing_image_paths(letter))

    first_incomplete_index = _find_first_incomplete_index(existing_summary, target_count)

    session = {
        "phase": "countdown",
        "letters": REC_LETTERS,
        "current_letter_index": first_incomplete_index,
        "accepted_vectors_current": [],
        "accepted_count_current": 0,
        "summary": existing_summary,
        "total_saved": int(sum(existing_summary.values())),
        "fps": int(fps),
        "similarity_threshold": float(similarity_threshold),
        "target_count": int(target_count),
        "countdown_seconds": 10,
        "pause_seconds": 5,
        "next_phase_at": None,
        "started_at": time.time(),
    }

    _activate_current_letter(session)
    return session


def _update_rec_phase(session):
    now = time.time()

    if session["phase"] == "countdown" and now >= session["next_phase_at"]:
        session["phase"] = "capturing"
        session["next_phase_at"] = None

    if session["phase"] == "pause" and now >= session["next_phase_at"]:
        next_index = session["current_letter_index"] + 1
        session["current_letter_index"] = next_index
        _activate_current_letter(session)


def _next_image_path(letter):
    letter_dir = os.path.join(CAPTURED_DIR, letter)
    existing_files = [
        file_name for file_name in os.listdir(letter_dir)
        if file_name.startswith(f"{letter}_") and file_name.lower().endswith(".png")
    ]
    next_index = len(existing_files) + 1
    return os.path.join(letter_dir, f"{letter}_{next_index:03d}.png")


def _compute_similarity_distance(flattened_vector, vectors):
    if not vectors:
        return None
    candidates = np.asarray(vectors, dtype=np.float32)
    distances = np.linalg.norm(candidates - flattened_vector.reshape(1, -1), axis=1)
    return float(np.min(distances))


def discover_model_versions(models_dir):
    version_pattern = re.compile(r"signalphaset_v(\d+)\.keras$")
    versions = []

    if not os.path.isdir(models_dir):
        return versions

    for name in os.listdir(models_dir):
        match = version_pattern.match(name)
        if match:
            versions.append(int(match.group(1)))

    return sorted(versions)


def get_model_path_by_version(model_version):
    return os.path.join(MODELS_DIR, f"signalphaset_v{model_version}.keras")

# -------------------------------------------------------------
# 2. LOAD RESOURCES & LABELS
# -------------------------------------------------------------
AVAILABLE_MODEL_VERSIONS = discover_model_versions(MODELS_DIR)
DEFAULT_MODEL_VERSION = AVAILABLE_MODEL_VERSIONS[-1] if AVAILABLE_MODEL_VERSIONS else None
model_cache = {}
model_runner_cache = {}
model_cache_lock = threading.Lock()


def _build_model_runner(model):
    @tf.function(reduce_retracing=True)
    def run_inference(inputs):
        return model(inputs, training=False)

    return run_inference


def load_model_for_version(model_version):
    cached_model = model_cache.get(model_version)
    if cached_model is not None:
        return cached_model

    with model_cache_lock:
        cached_model = model_cache.get(model_version)
        if cached_model is not None:
            return cached_model

        model_path = get_model_path_by_version(model_version)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model v{model_version} not found at {model_path}")

        loaded_model = tf.keras.models.load_model(model_path)
        model_cache[model_version] = loaded_model
        model_runner_cache[model_version] = _build_model_runner(loaded_model)
        logging.info(f"Loaded model v{model_version} from disk: {model_path}")
        return loaded_model


def get_model_runner_for_version(model_version):
    runner = model_runner_cache.get(model_version)
    if runner is not None:
        return runner

    load_model_for_version(model_version)
    return model_runner_cache[model_version]


if DEFAULT_MODEL_VERSION is not None:
    try:
        load_model_for_version(DEFAULT_MODEL_VERSION)
        logging.info(f"Default model loaded (v{DEFAULT_MODEL_VERSION}).")
    except Exception as e:
        logging.error(f"Could not load default model v{DEFAULT_MODEL_VERSION}: {e}")
else:
    logging.critical(f"No model versions found in {MODELS_DIR}")

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

MP_STATIC_IMAGE_MODE = True
MP_MAX_NUM_HANDS = 1
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MODEL_COMPLEXITY = 1

hands = mp_hands.Hands(
    static_image_mode=MP_STATIC_IMAGE_MODE,
    max_num_hands=MP_MAX_NUM_HANDS,
    min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
    model_complexity=MP_MODEL_COMPLEXITY
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


def calculate_extra_features(keypoints):
    wrist = keypoints[0]
    middle_finger_tip = keypoints[12]
    scale = np.linalg.norm(middle_finger_tip - wrist)
    if scale < 1e-8:
        scale = 1.0

    features = []
    for _, idx_a, idx_b in EXTRA_FEATURE_SPECS:
        dist = np.linalg.norm(keypoints[idx_a] - keypoints[idx_b])
        features.append(dist / scale)

    return np.asarray(features, dtype=np.float32)


def build_extra_feature_debug_data(keypoints):
    values = calculate_extra_features(keypoints)
    feature_lines = []

    for index, (name, idx_a, idx_b) in enumerate(EXTRA_FEATURE_SPECS):
        feature_lines.append({
            'name': name,
            'from': idx_a,
            'to': idx_b,
            'value': float(values[index]),
        })

    return feature_lines


def adapt_features_for_model(normalized_keypoints, model):
    model_input_shape = model.input_shape
    if isinstance(model_input_shape, list):
        model_input_shape = model_input_shape[0]

    keypoints_3d = np.asarray(normalized_keypoints, dtype=np.float32)
    flat_63 = keypoints_3d.reshape(-1)
    extra_features = calculate_extra_features(keypoints_3d)
    flat_73 = np.concatenate([flat_63, extra_features], axis=0).astype(np.float32)

    if len(model_input_shape) == 3:
        if model_input_shape[1:] != (NUM_LANDMARKS, COORD_DIMS):
            raise ValueError(f"Unsupported 3D model input shape: {model_input_shape}")
        return np.expand_dims(keypoints_3d, axis=0)

    if len(model_input_shape) == 2:
        expected_features = model_input_shape[1]
        if expected_features is None:
            raise ValueError("Model expects dynamic 2D feature size; not supported for live input.")

        if expected_features == BASE_KEYPOINT_FEATURES:
            features = flat_63
        elif expected_features == flat_73.shape[0]:
            features = flat_73
        elif expected_features < flat_73.shape[0]:
            features = flat_73[:expected_features]
        else:
            features = np.pad(flat_73, (0, expected_features - flat_73.shape[0]), mode="constant")

        return np.expand_dims(features.astype(np.float32), axis=0)

    raise ValueError(f"Unsupported model input rank for shape: {model_input_shape}")

def preprocess_keypoints(image, model, debug=False):
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

    # 4. Adapt features to selected model input
    input_data = adapt_features_for_model(normalized_keypoints, model)
    
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
    return render_template(
        'index.html',
        labels=labels,
        model_versions=AVAILABLE_MODEL_VERSIONS,
        default_model_version=DEFAULT_MODEL_VERSION,
    )


@app.route('/rec_data')
def rec_data_page():
    return render_template('rec_data.html')


@app.route('/rec_data/start', methods=['POST'])
def rec_data_start():
    global rec_session

    payload = request.json or {}
    fps = payload.get('fps', 30)
    similarity_threshold = payload.get('similarity_threshold', 0.12)
    target_count = payload.get('target_count', 100)

    try:
        fps = int(fps)
        similarity_threshold = float(similarity_threshold)
        target_count = int(target_count)
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid config values'}), 400

    if fps <= 0 or target_count <= 0 or similarity_threshold < 0:
        return jsonify({'error': 'fps/target_count must be > 0 and threshold >= 0'}), 400

    with rec_state_lock:
        rec_session = _new_rec_session(fps, similarity_threshold, target_count)
        state = _build_rec_state_payload(rec_session)

    return jsonify({'ok': True, 'state': state})


@app.route('/rec_data/stop', methods=['POST'])
def rec_data_stop():
    global rec_session

    with rec_state_lock:
        if rec_session is None:
            return jsonify({'ok': True, 'state': None})
        rec_session['phase'] = 'stopped'
        rec_session['next_phase_at'] = None
        state = _build_rec_state_payload(rec_session)

    return jsonify({'ok': True, 'state': state})


@app.route('/rec_data/state', methods=['GET'])
def rec_data_state():
    with rec_state_lock:
        if rec_session is None:
            return jsonify({'ok': True, 'state': None})
        _update_rec_phase(rec_session)
        state = _build_rec_state_payload(rec_session)
    return jsonify({'ok': True, 'state': state})


@app.route('/rec_data/frame', methods=['POST'])
def rec_data_frame():
    global rec_session

    data = request.json or {}
    if 'image' not in data:
        return jsonify({'error': 'No image received'}), 400

    with rec_state_lock:
        if rec_session is None:
            return jsonify({'error': 'No active session'}), 400

        _update_rec_phase(rec_session)
        if rec_session['phase'] != 'capturing':
            return jsonify({
                'accepted': False,
                'reason': f"phase_{rec_session['phase']}",
                'state': _build_rec_state_payload(rec_session)
            })

        current_letter = rec_session['letters'][rec_session['current_letter_index']]
        threshold = rec_session['similarity_threshold']

    try:
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'accepted': False, 'reason': 'decode_failed'}), 400

        keypoints, error, _ = extract_keypoints(image)
        if error:
            with rec_state_lock:
                state = _build_rec_state_payload(rec_session)
            return jsonify({'accepted': False, 'reason': 'no_hand', 'state': state})

        centered_keypoints = center_keypoints(keypoints)
        normalized_keypoints, _ = normalize_keypoints(centered_keypoints)
        flattened_vector = normalized_keypoints.reshape(-1).astype(np.float32)

        with rec_state_lock:
            min_distance = _compute_similarity_distance(
                flattened_vector,
                rec_session['accepted_vectors_current']
            )
            if min_distance is not None and min_distance < threshold:
                state = _build_rec_state_payload(rec_session)
                return jsonify({
                    'accepted': False,
                    'reason': 'too_similar',
                    'min_distance': min_distance,
                    'state': state,
                })

            image_path = _next_image_path(current_letter)
            saved = cv2.imwrite(image_path, image)
            if not saved:
                state = _build_rec_state_payload(rec_session)
                return jsonify({'accepted': False, 'reason': 'save_failed', 'state': state}), 500

            rec_session['accepted_vectors_current'].append(flattened_vector)
            rec_session['accepted_count_current'] += 1
            rec_session['summary'][current_letter] += 1
            rec_session['total_saved'] += 1

            if rec_session['accepted_count_current'] >= rec_session['target_count']:
                rec_session['phase'] = 'pause'
                rec_session['next_phase_at'] = time.time() + rec_session['pause_seconds']

            state = _build_rec_state_payload(rec_session)

        return jsonify({
            'accepted': True,
            'reason': 'saved',
            'min_distance': min_distance,
            'state': state,
        })

    except Exception as e:
        logging.error(f"rec_data frame error: {e}")
        with rec_state_lock:
            state = _build_rec_state_payload(rec_session) if rec_session is not None else None
        return jsonify({'accepted': False, 'reason': 'server_error', 'error': str(e), 'state': state}), 500


@app.route('/rec_data/summary', methods=['GET'])
def rec_data_summary():
    with rec_state_lock:
        if rec_session is None:
            return jsonify({'ok': True, 'summary': None})
        summary = {
            'per_letter': rec_session['summary'],
            'total_saved': rec_session['total_saved'],
            'dataset_path': CAPTURED_DIR,
            'zip_available': False,
        }
    return jsonify({'ok': True, 'summary': summary})

@app.route('/dataset_img/<label>')
def dataset_img(label):
    # Check for png first, then jpg
    for ext in ['.png', '.jpg']:
        filename = f"{label}{ext}"
        if os.path.exists(os.path.join(PICTURES_DIR, filename)):
             return send_from_directory(PICTURES_DIR, filename)
    return "", 404

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image received'}), 400

    model_version = data.get('model_version', DEFAULT_MODEL_VERSION)
    try:
        model_version = int(model_version)
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid model_version'}), 400

    if model_version not in AVAILABLE_MODEL_VERSIONS:
        return jsonify({
            'error': f'Model version v{model_version} not available',
            'available_versions': AVAILABLE_MODEL_VERSIONS,
        }), 400

    try:
        model = load_model_for_version(model_version)
        model_runner = get_model_runner_for_version(model_version)
    except Exception as e:
        logging.error(f"Could not load model v{model_version}: {e}")
        return jsonify({'error': f'Model v{model_version} could not be loaded'}), 500

    model_path = get_model_path_by_version(model_version)

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
        input_data, raw_kps, norm_kps, extra_info = preprocess_keypoints(image, model, debug=debug_mode)
        prep_time = (time.time() - prep_start) * 1000

        if input_data is None:
            return jsonify({'prediction': 'No Hand', 'confidence': 0.0})

        # 3. Prediction
        inference_start = time.time()
        prediction_tensor = model_runner(tf.convert_to_tensor(input_data, dtype=tf.float32))
        prediction = np.asarray(prediction_tensor)[0]
        inference_time = (time.time() - inference_start) * 1000
        
        predicted_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Get label from our fixed map
        predicted_label = idx_to_label.get(predicted_idx, f"Class {predicted_idx}")

        response = {
            'prediction': predicted_label,
            'confidence': confidence,
            'model_version': model_version,
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
                'mediapipe_confidence': f"{h_score*100:.1f}%",
                'mediapipe_label': h_label,
                'mediapipe_model_complexity': MP_MODEL_COMPLEXITY,
                'mediapipe_min_detection_confidence': MP_MIN_DETECTION_CONFIDENCE,
                'mediapipe_static_image_mode': MP_STATIC_IMAGE_MODE,
                'scale': f"{extra_info['scale_factor']:.4f}",
                'input_shape': str(extra_info['input_shape']),
                'model_version': model_version, 
                'model_path': model_path,
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
            
            debug_info = {
                'raw_landmarks': raw_kps.tolist(),
               # 'norm_landmarks': norm_kps.tolist(), # Skip sending this to save bandwidth
                'hand_cutout': f"data:image/jpeg;base64,{cutout_b64}"
            }

            if model_version == 3 and norm_kps is not None:
                debug_info['v3_feature_lines'] = build_extra_feature_debug_data(norm_kps)

            response['debug_info'] = debug_info

        return jsonify(response)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
