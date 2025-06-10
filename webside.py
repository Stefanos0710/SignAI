import os
from flask import jsonify, request, Flask, render_template
import shutil
import preprecessing_livedata_web as pre_data
import inference
from flask_socketio import SocketIO
import time

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

translated_word = ""

@app.route('/')
def index():
    return render_template("index.html", translated_word=translated_word)

@app.route('/clear-folder', methods=['POST'])
def clear_folder():
    folder_path = 'data/live/video'
    try:
        os.makedirs(folder_path, exist_ok=True)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Error: {e}')
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/save-video', methods=['POST'])
def save_video():
    try:
        video_file = request.files['video']
        if video_file:
            # Ensure directory exists
            os.makedirs('data/live/video', exist_ok=True)

            # Save the video file
            video_path = os.path.join('data/live/video', 'recorded_video.mp4')
            video_file.save(video_path)

            # Verify the file was saved
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                print(f"Video saved successfully at: {video_path}")
                print(f"Video size: {os.path.getsize(video_path)} bytes")

                # Start processing in background
                socketio.start_background_task(process_video)

                return jsonify({
                    'success': True,
                    'message': 'Video saved successfully',
                    'path': video_path
                })
            else:
                raise Exception("Video file was not saved properly")
    except Exception as e:
        print(f"Error saving video: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def process_video():
    try:
        global translated_word  # Wichtig: globale Variable verwenden

        # Preprocessing phase
        socketio.emit('processing_update', {
            'phase': 'preprocessing',
            'status': 'started',
            'message': 'Starting video preprocessing...'
        })

        pre_data.main()

        socketio.emit('processing_update', {
            'phase': 'preprocessing',
            'status': 'completed',
            'message': 'Preprocessing completed successfully!'
        })

        # Inference phase
        socketio.emit('processing_update', {
            'phase': 'inference',
            'status': 'started',
            'message': 'Starting sign language recognition...'
        })

        result = inference.main_inference("models/trained_model_v19.keras")

        # Speichern des übersetzten Wortes
        translated_word = result

        socketio.emit('processing_update', {
            'phase': 'inference',
            'status': 'completed',
            'message': 'Recognition completed!',
            'result': result
        })

        # Senden Sie ein zusätzliches Event für die Übersetzung
        socketio.emit('translation_update', {
            'translated_word': result
        })

    except Exception as e:
        print(f"Processing error: {str(e)}")
        socketio.emit('processing_update', {
            'phase': 'error',
            'status': 'error',
            'message': f'Error during processing: {str(e)}'
        })


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data/live/video', exist_ok=True)
    os.makedirs('data/live', exist_ok=True)

    print(f"\nAufnahme gestartet von {os.getenv('USERNAME', 'Unknown')}")
    print(f"Zeitstempel: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("Steuerung:")
    print("SPACE - Aufnahme starten/pausieren")
    print("Q     - Aufnahme beenden\n")

    # Add allow_unsafe_werkzeug=True to fix the Werkzeug warning
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
