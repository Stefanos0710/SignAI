"""
SignAI API Call Module

This module handles API calls to the SignAI translation service. Currently only for the desktop application.

## Metadata
- **Author**: Stefanos Koufogazos Loukianov
- **Original Creation Date**: 2025/10/11
- **Last Update**: 2025/12/11
"""
import sys
import os
import threading
import time
import requests
import socket
import psutil
import signal

# Add API folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from resource_path import resource_path

# Lazy placeholder
_signai_api = None
_signai_api_import_error = None

def _lazy_load_signai_api():
    """Lazy-load signai_api module to avoid early TensorFlow initialization."""
    global _signai_api, _signai_api_import_error
    if _signai_api or _signai_api_import_error:
        return _signai_api
    try:
        import api.signai_api as signai_api  # noqa: F401
        _signai_api = signai_api
        print('[API] signai_api loaded succsessfully (lazy).')
    except Exception as e:
        _signai_api_import_error = e
        print(f'[API] Error at launching signai_api (lazy): {e}')
    return _signai_api

class API:
    def __init__(self):
        # API configuration
        API_URL = "http://127.0.0.1:5000"
        self.API_UPLOAD_ENDPOINT = f"{API_URL}/api/upload"
        self.API_HEALTH_ENDPOINT = f"{API_URL}/api/health"

        # Instance variables to track if server is running
        self._server_running = False
        self._server_thread = None


    def is_port_in_use(self, port=5000):
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return False
            except socket.error:
                return True


    def get_process_using_port(self, port=5000):
        """Find the process that is using a specific port"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    try:
                        process = psutil.Process(conn.pid)
                        return {
                            'pid': conn.pid,
                            'name': process.name(),
                            'cmdline': ' '.join(process.cmdline())
                        }
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        return None
        except Exception as e:
            print(f"[Error] Error finding process on port {port}: {e}")
        return None


    def kill_process_on_port(self, port=5000):
        """Kill the process using the specified port"""
        process_info = self.get_process_using_port(port)

        if not process_info:
            print(f"✗ Could not find process using port {port}")
            return False

        pid = None
        try:
            pid = process_info['pid']
            name = process_info['name']

            print(f"→ Found process '{name}' (PID: {pid}) using port {port}")
            print(f"→ Killing process...")

            process = psutil.Process(pid)
            process.terminate()  # Try graceful termination first

            # Wait up to 3 seconds for process to terminate
            try:
                process.wait(timeout=3)
                print(f"✓ Process {pid} terminated successfully")
                time.sleep(0.5)  # Give OS time to release the port
                return True
            except psutil.TimeoutExpired:
                # Force kill if graceful termination didn't work
                print(f"→ Force killing process {pid}...")
                process.kill()
                process.wait(timeout=2)
                print(f"✓ Process {pid} killed")
                time.sleep(0.5)
                return True

        except psutil.NoSuchProcess:
            print(f"✓ Process already terminated")
            return True
        except psutil.AccessDenied:
            if pid:
                print(f"✗ Access denied - cannot kill process {pid}")
            else:
                print(f"✗ Access denied - cannot kill process (unknown PID)")
            print(f"  Run the application as administrator to kill this process")
            return False
        except Exception as e:
            print(f"✗ Error killing process: {e}")
            return False


    def start_api_server(self):
        if self._server_running:
            print("API server already running")
            return True

        # Stelle sicher, dass signai_api keine Site-Cleanup macht
        try:
            os.environ.setdefault('SIGNAI_DISABLE_SITE_CLEANUP', '1')
        except Exception:
            pass

        # Lazy load signai_api here
        api_mod = _lazy_load_signai_api()
        if not api_mod:
            print('[API] failed to start API server due to import error.')
            return False

        # Check if port is already in use
        if self.is_port_in_use(5000):
            print("[Error] Port 5000 is already in use!")
            print("Checking if it's our API server...")

            # Check if it's actually our API responding
            if self.check_api_health():
                print("✓ API server is already running on port 5000")
                self._server_running = True
                return True
            else:
                print("✗ Port 5000 is occupied by another application")
                print("→ Attempting to free port 5000...")

                if self.kill_process_on_port(5000):
                    print("✓ Port 5000 freed successfully")
                    # Verify port is now free
                    if not self.is_port_in_use(5000):
                        print("✓ Port is now available, starting API server...")
                    else:
                        print("✗ Port is still in use, cannot start API server")
                        return False
                else:
                    print("✗ Failed to free port 5000")
                    print("  Please close the application using port 5000 manually")
                    return False

        def run_server():
            try:
                self._server_running = True
                # Start Flask server
                api_mod.app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
            except Exception as e:
                print(f"✗ Error starting Flask server: {e}")
                self._server_running = False

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
            except requests.exceptions.ConnectionError:
                continue
            except Exception as e:
                print(f"[Error] Error checking API health: {e}")
                continue

        print("✗ Failed to start API server - timeout after 5 seconds")
        self._server_running = False
        return False


    def check_api_health(self):
        try:
            response = requests.get(self.API_HEALTH_ENDPOINT, timeout=1.5)
            return response.status_code == 200
        except Exception:
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
                response = requests.post(self.API_UPLOAD_ENDPOINT, files=files, timeout=120)

                # Debug: show raw response for troubleshooting
                try:
                    print(f"→ API HTTP {response.status_code}")
                    print(f"→ API raw response: {response.text}")
                except Exception:
                    pass

                # Parse response safely
                try:
                    result = response.json()
                except ValueError:
                    print("✗ API returned non-JSON response")
                    return {
                        "success": False,
                        "error": "Invalid response from API (not JSON)",
                        "raw_response": response.text,
                        "status_code": response.status_code
                    }

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
                    translation = result.get("translation")
                    if translation in (None, "", "<no translation>"):
                        confidence = result.get("confidence", 0)
                        top_preds = result.get("top_predictions", [])
                        print("✗ No valid translation returned by API")
                        print(f"  confidence={confidence}, top_predictions={top_preds}")
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
        test_video = resource_path("videos/current_video.mp4")

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
            result = {"success": False, "error": "Test video not found"}

        return result
