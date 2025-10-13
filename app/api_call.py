"""
SignAI API Call Module

This module handles API calls to the SignAI translation service.
"""
import sys
import os
import threading
import time
import requests

# Add API folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import api.signai_api as signai_api

class API:
    def __init__(self):
        # API configuration
        API_URL = "http://127.0.0.1:5000"
        self.API_UPLOAD_ENDPOINT = f"{API_URL}/api/upload"
        self.API_HEALTH_ENDPOINT = f"{API_URL}/api/health"

        # Instance variables to track if server is running
        self._server_running = False
        self._server_thread = None


    def start_api_server(self):
        if self._server_running:
            print("API server already running")
            return True

        def run_server():
            self._server_running = True

            # Start Flask server
            signai_api.app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

        # Start server in daemon thread
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        # Wait for server to start
        print("Starting API server...")
        for i in range(10):
            time.sleep(0.5)
            try:
                response = requests.get(self.API_HEALTH_ENDPOINT, timeout=1)
                if response.status_code == 200:
                    print("✓ API server is ready")
                    return True
            except:
                continue

        print("Failed to start API server")
        return False


    def check_api_health(self):
        try:
            response = requests.get(self.API_HEALTH_ENDPOINT, timeout=2)
            return response.status_code == 200
        except:
            return False


    def api_translation(self, video_path):
        """
        Send video to SignAI API for translation

        Args:
            video_path (str): Path to the video file to translate

        Returns:
            dict: Response containing:
                - success (bool): Whether translation was successful
                - translation (str): The predicted sign language word/gloss
                - confidence (float): Confidence percentage of the prediction
                - top_predictions (list): Top 5 predictions with confidence scores
                - model (str): Name of the model used
                - timing (dict): Detailed timing information
                - video_info (dict): Video file information
                - error (str): Error message if failed
        """
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"✗ Error: Video file not found: {video_path}")
            return {
                "success": False,
                "error": f"Video file not found: {video_path}"
            }

        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size < 1024:  # Less than 1KB
            print(f"✗ Error: Video file too small ({file_size} bytes)")
            return {
                "success": False,
                "error": f"Video file is too small ({file_size} bytes), likely empty or corrupted"
            }

        print(f"→ Sending video to API: {video_path} ({file_size / 1024:.1f} KB)")

        # Check if API server is running, start if not
        if not self.check_api_health():
            print("API server not running, starting it...")
            if not self.start_api_server():
                return {
                    "success": False,
                    "error": "Failed to start API server"
                }

        try:
            # Send video to API
            with open(video_path, "rb") as video_file:
                files = {"raw_video": (os.path.basename(video_path), video_file)}

                # Upload video to API
                print("→ Uploading video to API...")
                response = requests.post(self.API_UPLOAD_ENDPOINT, files=files, timeout=60)

                # Parse response
                result = response.json()

                if result.get("success"):
                    print(f"\n{'='*60}")
                    print("✓ Translation Successful!")
                    print(f"{'='*60}")
                    print(f"\nTranslation: {result.get('translation', 'N/A').upper()}")
                    print(f"Confidence: {result.get('confidence', 0)}%")

                    # Show top predictions
                    top_preds = result.get('top_predictions', [])
                    if top_preds:
                        print(f"\nTop 5 Predictions:")
                        for i, pred in enumerate(top_preds, 1):
                            word = pred.get('word', 'N/A').upper()
                            conf = pred.get('confidence', 0)
                            bar = '█' * int(conf / 5)  # Visual bar
                            print(f"  {i}. {word:<15} {conf:>6.2f}% {bar}")

                    # Show timing information
                    timing = result.get('timing', {})
                    if timing:
                        print(f"\nProcessing Time:")
                        print(f"  Total:          {timing.get('total_processing_time', 0)}s")
                        print(f"  Upload:         {timing.get('upload_time', 0)}s")
                        print(f"  Preprocessing:  {timing.get('preprocessing_time', 0)}s")
                        print(f"  Model Load:     {timing.get('model_load_time', 0)}s")
                        print(f"  Inference:      {timing.get('inference_time', 0)}s")

                    # Show video info
                    video_info = result.get('video_info', {})
                    if video_info:
                        print(f"\nVideo Info:")
                        print(f"  File: {video_info.get('filename', 'N/A')}")
                        print(f"  Size: {video_info.get('size_mb', 0)} MB")

                    print(f"\nModel: {result.get('model', 'N/A')}")
                    print(f"{'='*60}\n")
                else:
                    print(f"✗ Translation failed: {result.get('error')}")

                return result
        # Handle all errors
        except requests.exceptions.Timeout:
            error_msg = "API request timed out (video processing took too long)"
            print(f"✗ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to API server"
            print(f"✗ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Error during API call: {str(e)}"
            print(f"✗ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }


    def start(self):
        # Test with current video
        test_video = "videos/current_video.mp4"

        if os.path.exists(test_video):
            print(f"\n{'='*50}")
            print("Testing SignAI API Translation")
            print(f"{'='*50}\n")

            result = self.api_translation(test_video)

            print(f"\n{'='*50}")
            print("Result:")
            print(f"{'='*50}")
            print(f"Success: {result.get('success')}")
            if result.get('success'):
                print(f"Translation: {result.get('translation')}")
                print(f"Model: {result.get('model')}")
            else:
                print(f"Error: {result.get('error')}")
        else:
            print(f"Test video not found: {test_video}")
            print("Please record a video first using the desktop app")


if __name__ == "__main__":
    api = API()
    api.start()
