const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const resultDiv = document.getElementById('result');
const confDiv = document.getElementById('confidence');
const statusDiv = document.getElementById('status');

let intervalId = null;
let isProcessing = false;
let stream = null;

async function startCamera() {
    if (stream) return;

    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: 640, 
                height: 480,
                frameRate: { ideal: 30 }
            } 
        });
        video.srcObject = stream;
        statusDiv.innerText = "Camera active. Starting predictions...";
        
        // Wait for video to be ready
        video.onloadedmetadata = () => {
            intervalId = setInterval(processFrame, 10);
        };
    } catch (err) {
        console.error("Error accessing camera: ", err);
        statusDiv.innerText = "Error: Could not access camera. Please allow permissions.";
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }
    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }
    resultDiv.innerText = "Stopped";
    confDiv.innerText = "";
    statusDiv.innerText = "Camera stopped.";
}

function processFrame() {
    if (isProcessing) return; // Prevent stacking requests
    if (!stream || video.paused || video.ended) return;

    isProcessing = true;

    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to Base64 (JPEG, quality 0.7 for speed)
    const dataURL = canvas.toDataURL('image/jpeg', 0.7);

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error(data.error);
        } else {
            resultDiv.innerText = data.prediction;
            // Color code confidence
            const confPercent = Math.round(data.confidence * 100);
            confDiv.innerText = `Confidence: ${confPercent}%`;
            
            if (confPercent > 80) resultDiv.style.color = "#4CAF50"; // Green
            else if (confPercent > 50) resultDiv.style.color = "#FFC107"; // Yellow
            else resultDiv.style.color = "#FF5722"; // Red
        }
    })
    .catch(err => {
        console.error("Prediction error:", err);
    })
    .finally(() => {
        isProcessing = false;
    });
}