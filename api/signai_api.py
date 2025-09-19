"""
SignAI - Sign Language Translator
api Module

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

# Import pipeline modules
import preprocessing_live_data as pre
import inference

# Flask app instance (referenced by request.py)
app = Flask(__name__)

# Configuration
UPLOAD_DIR = os.path.join('../data', 'live', 'video')
CSV_DIR = os.path.join('../data', 'live')
MODEL_PATH = os.path.join('../models', 'trained_model_v21.keras')

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
        # Validate file in request
        if 'raw_video' not in request.files:
            return jsonify({'success': False, 'error': 'Kein Video im Feld "raw_video" gefunden.'}), 400

        file = request.files['raw_video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Leerer Dateiname.'}), 400

        # Save uploaded video
        filename = secure_filename(file.filename) or 'uploaded_video.mp4'
        # Force mp4 extension to keep VideoWriter/codec expectations simple
        if not filename.lower().endswith('.mp4'):
            filename = os.path.splitext(filename)[0] + '.mp4'

        saved_video_path = os.path.join(UPLOAD_DIR, 'recorded_video.mp4')
        file.save(saved_video_path)

        # Preprocess video to CSV (no UI windows for server mode)
        pre.main(video_path=saved_video_path, show_windows=False)

        # Run inference using the latest model
        predicted_word = inference.main_inference(MODEL_PATH)
        if not predicted_word:
            return jsonify({'success': False, 'error': 'Keine Vorhersage möglich.'}), 500

        # Build response (keep behavior close to normal AI translation)
        return jsonify({
            'success': True,
            'translation': predicted_word,
            'model': os.path.basename(MODEL_PATH)
        })

    except Exception as e:
        # Collect stack trace for server logs, return concise error to client
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# Optional direct run (normally started by request.py)
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
