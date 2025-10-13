# SignAI API - Function Reference Guide

**Version:** 1.0  
**Last Updated:** 2025-10-13  
**Author:** Stefanos Koufogazos Loukianov

---

## Overview

This guide explains all available functions in the SignAI API and how to use them in your own files.

---

## Table of Contents

1. [Module: `request.py`](#module-requestpy)
2. [Module: `signai_api.py`](#module-signai_apipy)
3. [Module: `preprocessing_live_data.py`](#module-preprocessing_live_datapy)
4. [Module: `inference.py`](#module-inferencepy)
5. [Complete Examples](#complete-examples)

---

## Module: `request.py`

### Function: `start(video_path)`

**Description:**  
Starts the SignAI API server in the background and sends a video for translation.

**Parameters:**
- `video_path` (str): Path to the video file to translate

**Returns:**  
None (prints response to console)

**Usage in your file:**

```python
import sys
sys.path.append('../api')  # Add API folder to path
import request

# Send video to API
request.start("path/to/video.mp4")
```

**Example:**

```python
# app/api_call.py
import sys
sys.path.append('../api')
import request

def translate_video(video_path):
    """Send video to SignAI API for translation"""
    request.start(video_path)

# Usage
translate_video("videos/current_video.mp4")
```

**Output:**
```
Response: {"success": true, "translation": "hello", "model": "trained_model_v21.keras"}
```

---

### Function: `run_api()`

**Description:**  
Internal function that runs the Flask API server.

**Parameters:** None

**Returns:** None

**Note:** This is called automatically by `start()`. You don't need to call it directly.

---

## Module: `signai_api.py`

### Flask App Instance: `app`

**Description:**  
The main Flask application instance.

**Usage in your file:**

```python
from api import signai_api

# Access the Flask app
app = signai_api.app

# Run the server manually
app.run(host='127.0.0.1', port=5000)
```

---

### API Endpoint: `GET /api/health`

**Description:**  
Health check endpoint to verify the API is running.

**Usage with requests:**

```python
import requests

response = requests.get("http://127.0.0.1:5000/api/health")
print(response.json())
# Output: {'status': 'ok', 'message': 'SignAI server API healthy'}
```

---

### API Endpoint: `POST /api/upload`

**Description:**  
Upload a video and get the sign language translation.

**Parameters (Form Data):**
- `raw_video` (file): The video file to translate

**Response:**
```json
{
    "success": true,
    "translation": "hello",
    "model": "trained_model_v21.keras"
}
```

**Usage with requests:**

```python
import requests

url = "http://127.0.0.1:5000/api/upload"

with open("video.mp4", "rb") as f:
    files = {"raw_video": ("video.mp4", f)}
    response = requests.post(url, files=files)
    
result = response.json()
if result["success"]:
    print(f"Translation: {result['translation']}")
else:
    print(f"Error: {result['error']}")
```

---

## Module: `preprocessing_live_data.py`

### Function: `main(video_path, show_windows=True)`

**Description:**  
Processes a video file to extract keypoints (face, hands, body) and saves them to CSV.

**Parameters:**
- `video_path` (str): Path to the input video file
- `show_windows` (bool, optional): Whether to show processing windows (default: True)

**Returns:**  
None (saves data to `../data/live/live_dataset.csv`)

**Output File:**  
CSV file with columns:
- `pose_{0-32}_x`, `pose_{0-32}_y` (33 body keypoints)
- `hand_{0-41}_x`, `hand_{0-41}_y` (42 hand keypoints)
- `face_{0-467}_x`, `face_{0-467}_y` (468 face keypoints)

**Usage in your file:**

```python
import sys
sys.path.append('../api')
import preprocessing_live_data as pre

# Process video without showing windows
pre.main(video_path="videos/my_video.mp4", show_windows=False)

# Process video with visual feedback
pre.main(video_path="videos/my_video.mp4", show_windows=True)
```

**Example:**

```python
# app/api_call.py
import sys
sys.path.append('../api')
import preprocessing_live_data as pre

def preprocess_video(video_path, show_ui=False):
    """Extract keypoints from video"""
    try:
        pre.main(video_path=video_path, show_windows=show_ui)
        print(f"✓ Video preprocessed: {video_path}")
        return True
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False

# Usage
success = preprocess_video("videos/recording.mp4")
```

---

## Module: `inference.py`

### Function: `main_inference(model_path)`

**Description:**  
Loads the trained model and predicts the sign language translation from preprocessed CSV data.

**Parameters:**
- `model_path` (str): Path to the trained Keras model (.keras file)

**Returns:**  
`str` - The predicted word/gloss, or `None` if prediction fails

**Required Files:**
- Model: `models/trained_model_v21.keras`
- Tokenizer: `tokenizers/gloss_tokenizer.json`
- CSV Data: `data/live/live_dataset.csv`

**Usage in your file:**

```python
import sys
sys.path.append('../api')
import inference

# Predict from preprocessed data
model_path = "../models/trained_model_v21.keras"
prediction = inference.main_inference(model_path)

if prediction:
    print(f"Predicted sign: {prediction}")
else:
    print("Prediction failed")
```

**Example:**

```python
# app/api_call.py
import sys
sys.path.append('../api')
import inference

def get_translation(model_path="../models/trained_model_v21.keras"):
    """Get translation from preprocessed video"""
    try:
        result = inference.main_inference(model_path)
        if result:
            print(f"Translation: {result}")
            return result
        else:
            print("No prediction available")
            return None
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

# Usage
translation = get_translation()
```

---

### Function: `load_tokenizer(tokenizer_path)`

**Description:**  
Loads the tokenizer from a JSON file.

**Parameters:**
- `tokenizer_path` (str): Path to the tokenizer JSON file

**Returns:**  
Keras tokenizer object

**Usage in your file:**

```python
import sys
sys.path.append('../api')
import inference

tokenizer = inference.load_tokenizer("../tokenizers/gloss_tokenizer.json")
print(f"Vocabulary size: {len(tokenizer.word_index)}")
```
 
---

### Function: `load_and_prepare_data(csv_file_path)`

**Description:**  
Loads keypoint data from CSV and prepares it for model input.

**Parameters:**
- `csv_file_path` (str): Path to the CSV file with keypoint data

**Returns:**  
`numpy.ndarray` - Normalized and reshaped data ready for model input

**Usage in your file:**

```python
import sys
sys.path.append('../api')
import inference

# Load and prepare data
data = inference.load_and_prepare_data("../data/live/live_dataset.csv")
print(f"Data shape: {data.shape}")
# Output: Data shape: (1, 1, 1086)
```

---

## Complete Examples

### Example 1: Simple Translation (Using request.py)

```python
# app/api_call.py
"""
Simple API call for sign language translation
"""
import sys
sys.path.append('../api')
import request

def translate_sign_language(video_path):
    """
    Translate a sign language video to text
    
    Args:
        video_path (str): Path to the video file
    """
    print(f"Translating video: {video_path}")
    request.start(video_path)

# Usage
if __name__ == "__main__":
    translate_sign_language("videos/current_video.mp4")
```

---

### Example 2: Step-by-Step Pipeline

```python
# app/api_call.py
"""
Manual step-by-step translation pipeline
"""
import sys
sys.path.append('../api')
import preprocessing_live_data as pre
import inference

def translate_video_manual(video_path, model_path="../models/trained_model_v21.keras"):
    """
    Translate video using manual pipeline
    
    Args:
        video_path (str): Path to input video
        model_path (str): Path to trained model
        
    Returns:
        str: Predicted translation or None
    """
    try:
        # Step 1: Preprocess video to extract keypoints
        print("Step 1: Extracting keypoints...")
        pre.main(video_path=video_path, show_windows=False)
        
        # Step 2: Run inference on extracted data
        print("Step 2: Running inference...")
        prediction = inference.main_inference(model_path)
        
        if prediction:
            print(f"✓ Translation: {prediction}")
            return prediction
        else:
            print("✗ No prediction available")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

# Usage
if __name__ == "__main__":
    result = translate_video_manual("videos/current_video.mp4")
```

---

### Example 3: Using HTTP Requests Directly

```python
# app/api_call.py
"""
Translate using HTTP requests to API
"""
import requests
import os

def translate_via_api(video_path):
    """
    Send video to API via HTTP request
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: API response or error
    """
    api_url = "http://127.0.0.1:5000/api/upload"
    
    # Check if video exists
    if not os.path.exists(video_path):
        return {"success": False, "error": "Video file not found"}
    
    try:
        # Send video to API
        with open(video_path, "rb") as f:
            files = {"raw_video": (os.path.basename(video_path), f)}
            response = requests.post(api_url, files=files)
        
        return response.json()
        
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "API server not running"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Usage
if __name__ == "__main__":
    result = translate_via_api("videos/current_video.mp4")
    
    if result.get("success"):
        print(f"Translation: {result['translation']}")
    else:
        print(f"Error: {result['error']}")
```

---

### Example 4: Integration with Desktop App

```python
# app/api_call.py
"""
Integration with SignAI desktop application
"""
import sys
import os
sys.path.append('../api')
import request
import threading

class SignAITranslator:
    def __init__(self):
        self.current_video_path = "videos/current_video.mp4"
        self.translation = None
        
    def translate_async(self, video_path=None):
        """
        Translate video asynchronously (non-blocking)
        
        Args:
            video_path (str, optional): Path to video, uses current_video if None
        """
        if video_path is None:
            video_path = self.current_video_path
            
        # Run translation in separate thread
        thread = threading.Thread(target=self._translate, args=(video_path,))
        thread.daemon = True
        thread.start()
        
    def _translate(self, video_path):
        """Internal translation method"""
        try:
            request.start(video_path)
        except Exception as e:
            print(f"Translation error: {e}")
    
    def translate_sync(self, video_path=None):
        """
        Translate video synchronously (blocking)
        
        Args:
            video_path (str, optional): Path to video
            
        Returns:
            str: Translation result
        """
        if video_path is None:
            video_path = self.current_video_path
            
        try:
            request.start(video_path)
        except Exception as e:
            print(f"Translation error: {e}")
            return None

# Usage in app.py
# from api_call import SignAITranslator
#
# translator = SignAITranslator()
# translator.translate_async("videos/recording.mp4")
```

---

### Example 5: Batch Processing Multiple Videos

```python
# app/api_call.py
"""
Process multiple videos in batch
"""
import sys
import os
sys.path.append('../api')
import preprocessing_live_data as pre
import inference

def batch_translate_videos(video_folder, model_path="../models/trained_model_v21.keras"):
    """
    Translate all videos in a folder
    
    Args:
        video_folder (str): Path to folder containing videos
        model_path (str): Path to trained model
        
    Returns:
        dict: Dictionary mapping video names to translations
    """
    results = {}
    
    # Get all video files
    video_files = [f for f in os.listdir(video_folder) 
                   if f.endswith(('.mp4', '.avi', '.mov'))]
    
    print(f"Processing {len(video_files)} videos...")
    
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        
        try:
            # Preprocess
            pre.main(video_path=video_path, show_windows=False)
            
            # Predict
            prediction = inference.main_inference(model_path)
            
            results[video_file] = prediction if prediction else "No prediction"
            print(f"✓ {video_file}: {results[video_file]}")
            
        except Exception as e:
            results[video_file] = f"Error: {str(e)}"
            print(f"✗ {video_file}: {results[video_file]}")
    
    return results

# Usage
if __name__ == "__main__":
    translations = batch_translate_videos("videos/history")
    
    # Print summary
    print("\n=== Translation Summary ===")
    for video, translation in translations.items():
        print(f"{video}: {translation}")
```

---

## Error Handling

### Common Errors and Solutions

**1. ModuleNotFoundError**
```python
# Problem: Cannot find API modules
# Solution: Add API folder to path
import sys
sys.path.append('../api')  # Adjust path as needed
```

**2. FileNotFoundError**
```python
# Problem: Model or tokenizer not found
# Solution: Check file paths
import os

model_path = "../models/trained_model_v21.keras"
if not os.path.exists(model_path):
    print(f"Model not found at: {model_path}")
```

**3. Connection Error**
```python
# Problem: API server not running
# Solution: Check if server is running
import requests

try:
    response = requests.get("http://127.0.0.1:5000/api/health")
    print("API is running")
except requests.exceptions.ConnectionError:
    print("API server is not running. Start it first!")
```

---

## Best Practices

1. **Always check file existence before processing:**
```python
import os

if os.path.exists(video_path):
    request.start(video_path)
else:
    print(f"Video not found: {video_path}")
```

2. **Use try-except for error handling:**
```python
try:
    prediction = inference.main_inference(model_path)
except Exception as e:
    print(f"Error: {e}")
    prediction = None
```

3. **Use relative paths from your current location:**
```python
import os

# Get absolute path from current file location
current_dir = os.path.dirname(__file__)
video_path = os.path.join(current_dir, "videos", "current_video.mp4")
```

4. **Close resources properly:**
```python
# When manually using preprocessing
try:
    pre.main(video_path, show_windows=False)
finally:
    # Cleanup happens automatically in the module
    pass
```

---

## Configuration

### Paths (configured in `signai_api.py`)

```python
# Video upload directory
UPLOAD_DIR = '../data/live/video'

# CSV output directory
CSV_DIR = '../data/live'

# Model path
MODEL_PATH = '../models/trained_model_v21.keras'

# Tokenizer path
TOKENIZER_PATH = '../tokenizers/gloss_tokenizer.json'
```

### Server Configuration

```python
# Default: localhost only
app.run(host='127.0.0.1', port=5000, debug=True)

# Network access (be careful!)
app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## Troubleshooting

### Video Processing Issues

```python
# Check if video is readable
import cv2

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Cannot open video")
else:
    print("Video is readable")
    cap.release()
```

### Model Loading Issues

```python
# Verify model file
import os

model_path = "../models/trained_model_v21.keras"
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model found: {size_mb:.2f} MB")
else:
    print("Model not found!")
```

---

## Additional Resources

- **Main Project README:** See root directory
- **Training Documentation:** See `train.py`
- **Model Architecture:** See `model.py`
- **Desktop App:** See `app/app.py`

---

## License

See LICENSE file in project root.

**Author:** Stefanos Koufogazos Loukianov  
**Project:** SignAI - Sign Language Translator

