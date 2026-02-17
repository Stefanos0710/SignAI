const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
// Optional: check if elements exist before using properties (in case of page load race)
const landmarksCanvas = document.getElementById('landmarks');
const landmarksCtx = landmarksCanvas ? landmarksCanvas.getContext('2d') : null;

const resultDiv = document.getElementById('result');
const confDiv = document.getElementById('confidence');
const statusDiv = document.getElementById('status');
const debugToggle = document.getElementById('debug-toggle');
const debugPanel = document.getElementById('debug-panel');
const handCutout = document.getElementById('hand-cutout');
const timingStats = document.getElementById('timing-stats');
const metaStats = document.getElementById('meta-stats');
const top5List = document.getElementById('top5-list');
const modelVersionSelect = document.getElementById('model-version');

let intervalId = null;
let isProcessing = false;
let stream = null;
let debugMode = false;

const container = document.getElementById('container');

function setCameraUIActive(isActive) {
    if (!container) return;
    container.classList.toggle('camera-inactive', !isActive);

    if (!isActive) {
        resultDiv.innerText = "";
        confDiv.innerText = "";
        if (landmarksCtx && landmarksCanvas) {
            landmarksCtx.clearRect(0, 0, landmarksCanvas.width, landmarksCanvas.height);
        }
    }
}

setCameraUIActive(false);

// Connections for Hand Landmarks (MediaPipe convention)
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index
    [9, 10], [10, 11], [11, 12],     // Middle
    [13, 14], [14, 15], [15, 16],    // Ring
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17], [0, 5], [0, 17] // Palm Base (Partial)
];

function toggleDebug() {
    if (!debugToggle) return;
    debugMode = debugToggle.checked;
    
    if (debugMode) {
        if (debugPanel) {
            debugPanel.style.display = "flex"; 
        }
        // Force loop restart to adjust interval if needed
        if (intervalId) {
            clearInterval(intervalId);
            intervalId = setInterval(processFrame, 60); // Upgrade to ~16 FPS for debug
        }
    } else {
        if (debugPanel) debugPanel.style.display = "none";
        if (landmarksCtx && landmarksCanvas) {
            landmarksCtx.clearRect(0, 0, landmarksCanvas.width, landmarksCanvas.height);
        }
        // Faster when debug is off
        if (intervalId) {
            clearInterval(intervalId);
            intervalId = setInterval(processFrame, 30); // ~30 FPS
        }
    }
}

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
            // Initial interval
            intervalId = setInterval(processFrame, 30);
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
    if (landmarksCtx && landmarksCanvas) {
        landmarksCtx.clearRect(0, 0, landmarksCanvas.width, landmarksCanvas.height);
    }
}

function drawLandmarks(landmarks) {
    if (!landmarksCtx || !landmarksCanvas) return;
    
    const width = landmarksCanvas.width;
    const height = landmarksCanvas.height;

    landmarksCtx.clearRect(0, 0, width, height);
    
    landmarksCtx.strokeStyle = "rgba(75, 92, 251, 0.8)"; // Primary color
    landmarksCtx.lineWidth = 2;
    landmarksCtx.lineCap = "round";

    // Draw connections
    landmarksCtx.beginPath();
    for (const [start, end] of HAND_CONNECTIONS) {
        const p1 = landmarks[start];
        const p2 = landmarks[end];
        if (p1 && p2) { // Ensure points exist
            landmarksCtx.moveTo(p1[0] * width, p1[1] * height);
            landmarksCtx.lineTo(p2[0] * width, p2[1] * height);
        }
    }
    landmarksCtx.stroke();

    // Draw points
    landmarksCtx.fillStyle = "#FF5722"; // Accent
    for (const point of landmarks) {
        landmarksCtx.beginPath();
        landmarksCtx.arc(point[0] * width, point[1] * height, 3, 0, 2 * Math.PI);
        landmarksCtx.fill();
    }
}

function renderTop5(top5data) {
    if (!top5List || !top5data) return;
    
    let html = '';
    top5data.forEach(item => {
        const percent = Math.round(item.confidence * 100);
        html += `
        <div class="top5-item">
            <div style="width: 20px; font-weight: bold;">${item.label}</div>
            <div class="bar-container">
                <div class="bar-fill" style="width: ${percent}%"></div>
            </div>
            <div class="prob-text">${percent}%</div>
        </div>
        `;
    });
    top5List.innerHTML = html;
}

function processFrame() {
    if (isProcessing) return; // Prevent stacking requests
    if (!stream || video.paused || video.ended) return;

    isProcessing = true;
    
    // Draw video frame to canvas
    // If debug is on, use lower quality JPEG to save bandwidth
    const quality = debugMode ? 0.5 : 0.7;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg', quality);

    // Prepare payload
    const payload = {
        image: dataURL,
        debug: debugMode,
        model_version: modelVersionSelect ? modelVersionSelect.value : null
    };

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error(data.error);
            if (statusDiv) statusDiv.innerText = `Error: ${data.error}`;
        } else {
            resultDiv.innerText = data.prediction;
            if (statusDiv && data.model_version !== undefined) {
                statusDiv.innerText = `Using model v${data.model_version}`;
            }
            // Color code confidence
            const confPercent = Math.round(data.confidence * 100);
            confDiv.innerText = `Confidence: ${confPercent}%`;
            
            if (confPercent > 80) resultDiv.style.color = "#4CAF50"; // Green
            else if (confPercent > 50) resultDiv.style.color = "#FFC107"; // Yellow
            else resultDiv.style.color = "#FF5722"; // Red

            // Handle Debug info
            if (debugMode && data.debug_info) {
                // 1. Hand Cutout
                if (handCutout && data.debug_info.hand_cutout) {
                    handCutout.src = data.debug_info.hand_cutout;
                }

                // 3. Landmarks
                if (data.debug_info.raw_landmarks) {
                    drawLandmarks(data.debug_info.raw_landmarks);
                }

                // 4. Timing
                if (data.timing && timingStats) {
                    timingStats.innerHTML = `
                        <li>Inference: <span>${data.timing.inference}</span></li>
                        <li>Preprocess: <span>${data.timing.preprocess}</span></li>
                        <li>Total: <span>${data.timing.total}</span></li>
                    `;
                }
                
                // 5. Hand Metadata
                if (data.meta && metaStats) {
                    metaStats.innerHTML = `
                        <li>Hand: <span>${data.meta.handedness}</span></li>
                        <li>Scale: <span>${data.meta.scale}</span></li>
                        <li>Shape: <span>${data.meta.input_shape}</span></li>
                    `;
                }

                // 6. Top 5
                if (data.top_5) {
                    renderTop5(data.top_5);
                }

            } else {
                // Clear landmarks if debug is off/no hand detected but debug is on
                if (landmarksCtx && landmarksCanvas) {
                    landmarksCtx.clearRect(0, 0, landmarksCanvas.width, landmarksCanvas.height);
                }
            }
        }
    })
    .catch(err => {
        console.error("Prediction error:", err);
    })
    .finally(() => {
        isProcessing = false;
    });
}
