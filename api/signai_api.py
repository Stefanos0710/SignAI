"""
SignAI - Sign Language Translator
API Module

This module implements the SignAI API server that accepts video uploads, preprocesses them,
and performs sign language translation using a trained model. It provides endpoints for health checks
and video translation.

Author: Stefanos Koufogazos Loukianov
Created: 2025-09-18
Updated: 2025-09-19 22:11
"""

from flask import Flask, request, jsonify
import os
import traceback
from werkzeug.utils import secure_filename
import time

# Import pipeline modules
from . import preprocessing_live_data as pre
from . import inference

try:
    from app.resource_path import resource_path, writable_path
except Exception:
    # Fallback implementations (minimal)
    import sys
    def resource_path(relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath('.'), relative_path)
    def writable_path(relative_path):
        if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.abspath('.')
        full = os.path.join(base, relative_path)
        parent = os.path.dirname(full)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return full

# Flask app instance (referenced by request.py)
app = Flask(__name__)

# Configuration
UPLOAD_DIR = writable_path(os.path.join('data', 'live', 'video'))
CSV_DIR = writable_path(os.path.join('data', 'live'))

model_candidate = None
try:
    candidate = resource_path(os.path.join('models', 'trained_model_v21.keras'))
    if os.path.exists(candidate):
        model_candidate = candidate
except Exception:
    model_candidate = None

if model_candidate:
    MODEL_PATH = model_candidate
else:
    # fallback to repo-location relative to this file
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model_v21.keras'))

# Ensure required directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'message': 'SignAI server API healthy'
    })


@app.route('/api/upload', methods=['POST'])
def upload_and_translate():
    try:
        request_start_time = time.time()

        # Validate file in request
        if 'raw_video' not in request.files:
            return jsonify({'success': False, 'error': 'No video found in field "raw_video".'}), 400

        file = request.files['raw_video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename.'}), 400

        # Save uploaded video
        upload_start = time.time()
        filename = secure_filename(file.filename) or 'uploaded_video.mp4'
        # Force mp4 extension to keep VideoWriter/codec expectations simple
        if not filename.lower().endswith('.mp4'):
            filename = os.path.splitext(filename)[0] + '.mp4'

        saved_video_path = os.path.join(UPLOAD_DIR, 'recorded_video.mp4')

        # DELETE OLD FILES BEFORE PROCESSING NEW VIDEO
        # Delete old video file if exists
        if os.path.exists(saved_video_path):
            try:
                os.remove(saved_video_path)
                print(f"✓ Deleted old video: {saved_video_path}")
            except Exception as e:
                print(f"[!] Could not delete old video: {e}")

        # Delete old CSV file if exists
        csv_output_path = os.path.join(CSV_DIR, 'live_dataset.csv')
        if os.path.exists(csv_output_path):
            try:
                os.remove(csv_output_path)
                print(f"✓ Deleted old CSV: {csv_output_path}")
            except Exception as e:
                print(f"[!] Could not delete old CSV: {e}")

        # Save the new video file
        file.save(saved_video_path)

        # Verify video was saved correctly
        if not os.path.exists(saved_video_path) or os.path.getsize(saved_video_path) == 0:
            return jsonify({'success': False, 'error': 'Failed to save video file'}), 500
        upload_time = time.time() - upload_start

        # Get video file size
        video_size_bytes = os.path.getsize(saved_video_path)
        video_size_mb = round(video_size_bytes / (1024 * 1024), 2)

        # Preprocess video to CSV (no UI windows for server mode)
        preprocessing_start = time.time()
        pre.main(video_path=saved_video_path, show_windows=False)
        preprocessing_time = time.time() - preprocessing_start

        # Run inference using the latest model
        inference_result = inference.main_inference(MODEL_PATH)

        if not inference_result or not inference_result.get('predicted_word'):
            return jsonify({'success': False, 'error': 'No prediction possible.'}), 500

        # Calculate total processing time
        total_processing_time = time.time() - request_start_time

        # Build detailed response
        return jsonify({
            'success': True,
            'translation': inference_result['predicted_word'],
            'confidence': inference_result['confidence'],
            'top_predictions': inference_result['top_predictions'],
            'model': os.path.basename(MODEL_PATH),
            'timing': {
                'total_processing_time': round(total_processing_time, 3),
                'upload_time': round(upload_time, 3),
                'preprocessing_time': round(preprocessing_time, 3),
                'model_load_time': inference_result['timing']['model_load_time'],
                'build_time': inference_result['timing']['build_time'],
                'data_load_time': inference_result['timing']['data_load_time'],
                'inference_time': inference_result['timing']['inference_time']
            },
            'video_info': {
                'filename': filename,
                'size_bytes': video_size_bytes,
                'size_mb': video_size_mb
            }
        })

    except Exception as e:
        # Collect stack trace for server logs, return concise error to client
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# Optional direct run (normally started by request.py)
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
