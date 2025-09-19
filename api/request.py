"""
SignAI - Sign Language Translator
request Module

This module starts a local SignAI API server and sends a video for processing.
The function `start(video_path)` launches the server in the background, uploads the specified video,
and prints the server's response.

Usage:
    import request
    request.start("video.mp4")

Author: Stefanos Koufogazos Loukianov
Created: 2025-09-19
Updated: 2025-09-19 22:07
"""

import threading
import time
import requests
import signai_api

# URL to the API server
url = "http://localhost:5000/api/upload"

# start API server
def run_api():
    signai_api.app.run(debug=True, use_reloader=False)

def start(video_path):
    thread = threading.Thread(target=run_api)
    thread.daemon = True
    thread.start()

    # short break to let the API server start
    time.sleep(2)

    # uploade video file
    with open(video_path, "rb") as f:
        video = {"raw_video": (video_path, f)}
        response = requests.post(url, files=video)
        print(f"Response: {response.text}")

