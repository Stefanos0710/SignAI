import os
import time
import shutil
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

# Import your custom modules
import preprecessing_livedata_web as pre_data
from api import inference

# Initialize Flask and SocketIO for local testing with threading
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

translated_word = ""  # Global variable to store the translation result


# Route for main page
@app.route('/')
def index():
    return render_template("index.html", translated_word=translated_word)

@app.route('/states')
def states():
    return render_template("states.html")

@app.route('/dev')
def dev():
    return render_template("dev.html")

# API endpoint to clear uploaded videos
@app.route('/clear-folder', methods=['POST'])
def clear_folder():
    folder_path = 'data/live/video'
    try:
        os.makedirs(folder_path, exist_ok=True)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# API endpoint to save the uploaded video
@app.route('/save-video', methods=['POST'])
def save_video():
    try:
        video_file = request.files['video']
        if video_file:
            os.makedirs('data/live/video', exist_ok=True)
            video_path = os.path.join('data/live/video', 'recorded_video.mp4')
            video_file.save(video_path)

            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                print(f"Video saved: {video_path}")
                print(f"Size: {os.path.getsize(video_path)} Bytes")

                # Start processing the video in a background task
                socketio.start_background_task(process_video)

                return jsonify({
                    'success': True,
                    'message': 'Video saved successfully',
                    'path': video_path
                })
            else:
                raise Exception("File was not saved correctly")
    except Exception as e:
        print(f"Error while saving video: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# Background function to process uploaded video
def process_video():
    global translated_word
    try:
        # Phase 1: Preprocessing
        socketio.emit('processing_update', {
            'phase': 'preprocessing',
            'status': 'started',
            'message': 'Preprocessing started...'
        })

        pre_data.main()

        socketio.emit('processing_update', {
            'phase': 'preprocessing',
            'status': 'completed',
            'message': '✅ Preprocessing completed'
        })

        # Phase 2: Inference
        socketio.emit('processing_update', {
            'phase': 'inference',
            'status': 'started',
            'message': 'Running inference...'
        })

        result = inference.main_inference("models/trained_model_v19.keras")
        if result == "haus":
            result = "house"
        elif result == "hund":
            result = "dog"
        elif result == "essen":
            result = "food"
        else:
            result = "tree"

        translated_word = result

        socketio.emit('processing_update', {
            'phase': 'inference',
            'status': 'completed',
            'message': '✅ Inference completed',
            'result': result
        })

        socketio.emit('translation_update', {
            'translated_word': result
        })

    except Exception as e:
        print(f"⚠️ Error during processing: {str(e)}")
        socketio.emit('processing_update', {
            'phase': 'error',
            'status': 'error',
            'message': f'Error: {str(e)}'
        })


# WebSocket connection events
@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


# Run the server locally
if __name__ == '__main__':
    # Make sure necessary folders exist
    os.makedirs('data/live/video', exist_ok=True)
    os.makedirs('data/live', exist_ok=True)

    print(f"\nRecording started by {os.getenv('USERNAME', 'Unknown')}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Controls:\nSPACE - Start/Pause\nQ - Stop\n")

    # Port for localhost
    port = int(os.environ.get('PORT', 8000))

    # Run the app on localhost
    socketio.run(
        app,
        host='127.0.0.1',
        port=port,
        debug=True,
        use_reloader=True,
        allow_unsafe_werkzeug=True
    )
