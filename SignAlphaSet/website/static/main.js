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
const cameraStats = document.getElementById('camera-stats');
const systemStats = document.getElementById('system-stats');
const predictionStats = document.getElementById('prediction-stats');
const debugJson = document.getElementById('debug-json');
const dbgFps = document.getElementById('dbg-fps');
const dbgRoundtrip = document.getElementById('dbg-roundtrip');
const dbgFrameCount = document.getElementById('dbg-frame-count');
const dbgLastUpdate = document.getElementById('dbg-last-update');

let intervalId = null;
let isProcessing = false;
let stream = null;
let debugMode = false;
let debugFrameCount = 0;
let lastFrameTs = null;
let smoothedFps = 0;
let isFullscreen = false;

const container = document.getElementById('container');
const fullscreenBtn = document.getElementById('fullscreen-btn');

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
        debugFrameCount = 0;
        lastFrameTs = null;
        smoothedFps = 0;
        if (dbgFps) dbgFps.textContent = '-';
        if (dbgRoundtrip) dbgRoundtrip.textContent = '-';
        if (dbgFrameCount) dbgFrameCount.textContent = '0';
        if (dbgLastUpdate) dbgLastUpdate.textContent = '-';
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

function formatMs(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "-";
    return `${Math.round(Number(value))} ms`;
}

function formatTime(value) {
    if (!value) return "-";
    return new Date(value).toLocaleTimeString();
}

function estimateKbFromDataUrl(dataURL) {
    if (!dataURL || typeof dataURL !== 'string') return 0;
    const base64 = dataURL.split(',')[1] || '';
    return Math.round((base64.length * 3) / 4 / 1024);
}

function updateDebugSummary(roundtripMs) {
    const now = performance.now();
    if (lastFrameTs !== null) {
        const delta = now - lastFrameTs;
        if (delta > 0) {
            const instantFps = 1000 / delta;
            smoothedFps = smoothedFps === 0 ? instantFps : (smoothedFps * 0.8 + instantFps * 0.2);
        }
    }
    lastFrameTs = now;
    debugFrameCount += 1;

    if (dbgFps) dbgFps.textContent = smoothedFps ? `${smoothedFps.toFixed(1)}` : '-';
    if (dbgRoundtrip) dbgRoundtrip.textContent = formatMs(roundtripMs);
    if (dbgFrameCount) dbgFrameCount.textContent = `${debugFrameCount}`;
    if (dbgLastUpdate) dbgLastUpdate.textContent = formatTime(Date.now());
}

function renderDebugData(data, contextInfo) {
    if (!debugMode) return;

    const top5 = Array.isArray(data.top_5) ? data.top_5 : [];
    const confPercent = data.confidence !== undefined ? Math.round(Number(data.confidence) * 100) : null;
    const top1 = top5[0];
    const top2 = top5[1];
    const topGap = top1 && top2 ? Math.round((top1.confidence - top2.confidence) * 100) : null;
    const landmarks = data.debug_info && data.debug_info.raw_landmarks ? data.debug_info.raw_landmarks : [];

    if (cameraStats) {
        cameraStats.innerHTML = `
            <li>State: <span>${stream ? 'active' : 'inactive'}</span></li>
            <li>Resolution: <span>${video.videoWidth || 0}x${video.videoHeight || 0}</span></li>
            <li>JPEG Quality: <span>${Math.round(contextInfo.quality * 100)}%</span></li>
            <li>Payload Size: <span>${contextInfo.payloadKb} KB</span></li>
            <li>Landmarks: <span>${landmarks.length}</span></li>
        `;
    }

    if (systemStats) {
        systemStats.innerHTML = `
            <li>Loop Delay: <span>${debugMode ? '60ms' : '30ms'}</span></li>
            <li>Canvas: <span>${canvas.width}x${canvas.height}</span></li>
            <li>Processing: <span>${isProcessing ? 'busy' : 'idle'}</span></li>
            <li>Model Selected: <span>v${modelVersionSelect ? modelVersionSelect.value : '-'}</span></li>
            <li>Model Active: <span>v${data.model_version ?? '-'}</span></li>
        `;
    }

    if (predictionStats) {
        predictionStats.innerHTML = `
            <li>Prediction: <span>${data.prediction ?? '-'}</span></li>
            <li>Confidence: <span>${confPercent !== null ? `${confPercent}%` : '-'}</span></li>
            <li>Top1/Top2 Gap: <span>${topGap !== null ? `${topGap}%` : '-'}</span></li>
            <li>Request RTT: <span>${formatMs(contextInfo.roundtripMs)}</span></li>
            <li>Status: <span>${data.error ? 'error' : 'ok'}</span></li>
        `;
    }

    if (debugJson) {
        const rawPreview = {
            prediction: data.prediction,
            confidence: data.confidence,
            model_version: data.model_version,
            timing: data.timing,
            meta: data.meta,
            top_5: top5.slice(0, 5)
        };
        debugJson.textContent = JSON.stringify(rawPreview, null, 2);
    }
}

async function startCamera() {
    if (stream) return;

    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1920 }, 
                height: { ideal: 1080 },
                frameRate: { ideal: 30 }
            } 
        });
        video.srcObject = stream;
        statusDiv.innerText = "Camera active. Starting predictions...";
        
        // Wait for video to be ready
        video.onloadedmetadata = () => {
            // Sync canvas sizes to actual video resolution
            const vw = video.videoWidth || 1920;
            const vh = video.videoHeight || 1080;
            canvas.width = vw;
            canvas.height = vh;
            landmarksCanvas.width = vw;
            landmarksCanvas.height = vh;

            // Initial interval
            intervalId = setInterval(processFrame, 30);
        };
    } catch (err) {
        console.error("Error accessing camera: ", err);
        statusDiv.innerText = "Error: Could not access camera. Please allow permissions.";
        setCameraUIActive(false);
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
    resultDiv.innerText = "";
    confDiv.innerText = "";
    statusDiv.innerText = "Camera stopped.";
    setCameraUIActive(false);
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
            <div class="top5-label">${item.label}</div>
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
    // Use high quality JPEG for best prediction accuracy
    const quality = debugMode ? 0.7 : 0.92;
    const requestStart = performance.now();

    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg', quality);
    const payloadKb = estimateKbFromDataUrl(dataURL);

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
        const roundtripMs = performance.now() - requestStart;
        if (debugMode) {
            updateDebugSummary(roundtripMs);
        }

        if (data.error) {
            console.error(data.error);
            if (statusDiv) statusDiv.innerText = `Error: ${data.error}`;
        } else {
            resultDiv.innerText = data.prediction;
            setCameraUIActive(true);
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

                renderDebugData(data, { roundtripMs, quality, payloadKb });

            } else {
                // Clear landmarks if debug is off/no hand detected but debug is on
                if (landmarksCtx && landmarksCanvas) {
                    landmarksCtx.clearRect(0, 0, landmarksCanvas.width, landmarksCanvas.height);
                }

                if (debugMode) {
                    renderTop5(Array.isArray(data.top_5) ? data.top_5 : []);
                    renderDebugData(data, { roundtripMs, quality, payloadKb });
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

// ===== FULLSCREEN MODE =====
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().then(() => {
            enterFullscreenMode();
        }).catch(err => {
            // Fallback: use CSS-only fullscreen if API fails
            enterFullscreenMode();
        });
    } else {
        document.exitFullscreen().then(() => {
            exitFullscreenMode();
        }).catch(() => {
            exitFullscreenMode();
        });
    }
}

function enterFullscreenMode() {
    isFullscreen = true;
    document.body.classList.add('fullscreen-mode');
    if (debugMode) {
        document.body.classList.add('debug-open');
    }
    if (fullscreenBtn) fullscreenBtn.textContent = 'Exit Fullscreen';
    // Collapse settings by default in fullscreen
    const controlsPanel = document.getElementById('controls-panel');
    const settingsBtn = document.getElementById('settings-toggle');
    if (controlsPanel) controlsPanel.classList.add('collapsed');
    if (settingsBtn) settingsBtn.classList.remove('active');
}

function exitFullscreenMode() {
    isFullscreen = false;
    document.body.classList.remove('fullscreen-mode', 'debug-open');
    if (fullscreenBtn) fullscreenBtn.textContent = 'Fullscreen';
    // Always show settings when leaving fullscreen
    const controlsPanel = document.getElementById('controls-panel');
    const settingsBtn = document.getElementById('settings-toggle');
    if (controlsPanel) controlsPanel.classList.remove('collapsed');
    if (settingsBtn) settingsBtn.classList.remove('active');
}

// Sync when user presses Escape or browser exits fullscreen
document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement && isFullscreen) {
        exitFullscreenMode();
    }
});

// Keep debug-open class in sync with debug toggle
const _origToggleDebug = toggleDebug;
toggleDebug = function() {
    _origToggleDebug();
    if (isFullscreen) {
        if (debugMode) {
            document.body.classList.add('debug-open');
        } else {
            document.body.classList.remove('debug-open');
        }
    }
};

// ===== COLLAPSIBLE SETTINGS IN FULLSCREEN =====
function toggleSettings() {
    const controlsPanel = document.getElementById('controls-panel');
    const settingsBtn = document.getElementById('settings-toggle');
    if (!controlsPanel) return;
    controlsPanel.classList.toggle('collapsed');
    if (settingsBtn) settingsBtn.classList.toggle('active');
}
