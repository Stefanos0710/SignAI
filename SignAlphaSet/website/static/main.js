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
const predictionStats = document.getElementById('prediction-stats');
const dbgFps = document.getElementById('dbg-fps');
const dbgRoundtrip = document.getElementById('dbg-roundtrip');
const toggleV3Features = document.getElementById('toggle-v3-features');

const DEBUG_KEYPOINT_RADIUS = 6;
const NORMAL_KEYPOINT_RADIUS = 3;

let intervalId = null;
let isProcessing = false;
let stream = null;
let debugMode = false;
let debugFrameCount = 0;
let lastFrameTs = null;
let smoothedFps = 0;
let isFullscreen = false;
let showV3FeatureOverlay = true;

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

function localizePredictionLabel(label) {
    if (!label) return label;
    return label === 'No Hand' ? t('noHand') : label;
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
}

function renderDebugData(data, contextInfo) {
    if (!debugMode) return;

    const confPercent = data.confidence !== undefined ? Math.round(Number(data.confidence) * 100) : null;
    const top5 = Array.isArray(data.top_5) ? data.top_5 : [];
    const top1 = top5[0] || null;
    const top2 = top5[1] || null;
    const topGap = top1 && top2 ? Math.round((top1.confidence - top2.confidence) * 100) : null;

    if (predictionStats) {
        const localizedPrediction = localizePredictionLabel(data.prediction ?? '-');
        const top1Label = top1 ? localizePredictionLabel(top1.label) : '-';
        const top2Label = top2 ? localizePredictionLabel(top2.label) : '-';
        const top1Conf = top1 ? `${Math.round(top1.confidence * 100)}%` : '-';
        const top2Conf = top2 ? `${Math.round(top2.confidence * 100)}%` : '-';
        predictionStats.innerHTML = `
            <li>${t('prediction')}: <span>${localizedPrediction}</span></li>
            <li>${t('confidence')}: <span>${confPercent !== null ? `${confPercent}%` : '-'}</span></li>
            <li>${t('top1Top2Gap')}: <span>${topGap !== null ? `${topGap}%` : '-'}</span></li>
            <li>${t('requestRtt')}: <span>${formatMs(contextInfo.roundtripMs)}</span></li>
        `;
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
        statusDiv.innerText = t('cameraActive');
        
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
        statusDiv.innerText = t('cameraError');
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
    statusDiv.innerText = t('cameraStopped');
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
    const pointRadius = debugMode ? DEBUG_KEYPOINT_RADIUS : NORMAL_KEYPOINT_RADIUS;
    for (const point of landmarks) {
        landmarksCtx.beginPath();
        landmarksCtx.arc(point[0] * width, point[1] * height, pointRadius, 0, 2 * Math.PI);
        landmarksCtx.fill();
    }
}

function drawV3FeatureLines(landmarks, featureLines) {
    if (!landmarksCtx || !landmarksCanvas) return;
    if (!Array.isArray(landmarks) || !Array.isArray(featureLines) || featureLines.length === 0) return;

    const width = landmarksCanvas.width;
    const height = landmarksCanvas.height;

    landmarksCtx.save();
    landmarksCtx.strokeStyle = 'rgba(255, 193, 7, 0.95)';
    landmarksCtx.fillStyle = 'rgba(255, 193, 7, 0.95)';
    landmarksCtx.lineWidth = 3;
    landmarksCtx.font = 'bold 12px Arial';

    for (const line of featureLines) {
        const fromIdx = Number(line.from);
        const toIdx = Number(line.to);
        const p1 = landmarks[fromIdx];
        const p2 = landmarks[toIdx];
        if (!p1 || !p2) continue;

        const x1 = p1[0] * width;
        const y1 = p1[1] * height;
        const x2 = p2[0] * width;
        const y2 = p2[1] * height;

        landmarksCtx.beginPath();
        landmarksCtx.moveTo(x1, y1);
        landmarksCtx.lineTo(x2, y2);
        landmarksCtx.stroke();

        const midX = (x1 + x2) / 2;
        const midY = (y1 + y2) / 2;
        const value = Number(line.value);
        if (!Number.isNaN(value)) {
            landmarksCtx.fillText(value.toFixed(2), midX + 6, midY - 6);
        }
    }

    landmarksCtx.restore();
}

function renderTop5(top5data) {
    if (!top5List || !top5data) return;
    
    let html = '';
    top5data.forEach(item => {
        const percent = Math.round(item.confidence * 100);
        const localizedLabel = localizePredictionLabel(item.label);
        html += `
        <div class="top5-item">
            <div class="top5-label">${localizedLabel}</div>
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
            if (statusDiv) statusDiv.innerText = `${t('error')}: ${data.error}`;
        } else {
            resultDiv.innerText = localizePredictionLabel(data.prediction);
            setCameraUIActive(true);
            if (statusDiv && data.model_version !== undefined) {
                statusDiv.innerText = `${t('usingModel')} v${data.model_version}`;
            }
            // Color code confidence
            const confPercent = Math.round(data.confidence * 100);
            confDiv.innerText = `${t('confidence')}: ${confPercent}%`;
            
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

                    if (
                        showV3FeatureOverlay &&
                        Number(data.model_version) === 3 &&
                        Array.isArray(data.debug_info.v3_feature_lines)
                    ) {
                        drawV3FeatureLines(data.debug_info.raw_landmarks, data.debug_info.v3_feature_lines);
                    }
                }

                // 4. Timing
                if (data.timing && timingStats) {
                    timingStats.innerHTML = `
                        <li>${t('decode')}: <span>${data.timing.decode ?? '-'}</span></li>
                        <li>${t('inference')}: <span>${data.timing.inference}</span></li>
                        <li>${t('preprocess')}: <span>${data.timing.preprocess}</span></li>
                        <li>${t('total')}: <span>${data.timing.total}</span></li>
                    `;
                }
                
                // 5. Hand Metadata
                if (data.meta && metaStats) {
                    metaStats.innerHTML = `
                        <li>${t('hand')}: <span>${data.meta.mediapipe_label ?? data.meta.handedness ?? '-'}</span></li>
                        <li>${t('mediapipeAccuracy')}: <span>${data.meta.mediapipe_confidence ?? '-'}</span></li>
                        <li>${t('mediapipeMinDetection')}: <span>${data.meta.mediapipe_min_detection_confidence ?? '-'}</span></li>
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
    if (fullscreenBtn) fullscreenBtn.textContent = t('exitFullscreen');
    // Collapse settings by default in fullscreen
    const controlsPanel = document.getElementById('controls-panel');
    const settingsBtn = document.getElementById('settings-toggle');
    if (controlsPanel) controlsPanel.classList.add('collapsed');
    if (settingsBtn) settingsBtn.classList.remove('active');
}

function exitFullscreenMode() {
    isFullscreen = false;
    document.body.classList.remove('fullscreen-mode', 'debug-open');
    if (fullscreenBtn) fullscreenBtn.textContent = t('fullscreen');
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

// ===== DICTIONARY OVERLAY =====
function toggleDictionaryOverlay() {
    const overlay = document.getElementById('dictionary-overlay');
    const dictBtn = document.getElementById('dictionary-toggle');
    if (!overlay) return;
    overlay.classList.toggle('visible');
    if (dictBtn) dictBtn.classList.toggle('active');
}

// ===== LANGUAGE SYSTEM =====
let currentLang = 'en';

const translations = {
    en: {
        pageTitle: 'SignAI Live Demo',
        startCamera: 'Start Camera',
        stopCamera: 'Stop Camera',
        fullscreen: 'Fullscreen',
        exitFullscreen: 'Exit Fullscreen',
        languageTitle: 'Language',
        settingsTitle: 'Settings',
        dictionaryTitle: 'Sign Dictionary',
        closeTitle: 'Close',
        modelVersion: 'Model Version:',
        debugMode: 'Debug Mode',
        waiting: 'Waiting...',
        ready: 'Ready',
        dictTitle: 'Sign Language Dictionary',
        debugCenter: 'Debug Center',
        roundtrip: 'Roundtrip',
        frames: 'Frames',
        lastUpdate: 'Last Update',
        handCutout: 'Hand Cutout',
        cameraStream: 'Camera / Stream',
        waitingCamera: 'Waiting for camera...',
        performance: 'Performance',
        waitingData: 'Waiting for data...',
        system: 'System',
        prediction: 'Prediction',
        handMeta: 'Hand Meta',
        top5Predictions: 'Top 5 Predictions',
        rawResponse: 'Raw Response (Live)',
        waitingDebugData: 'Waiting for debug data...',
        noHand: 'No Hand',
        cameraActive: 'Camera active. Starting predictions...',
        cameraError: 'Error: Could not access camera. Please allow permissions.',
        cameraStopped: 'Camera stopped.',
        usingModel: 'Using model',
        confidence: 'Confidence',
        state: 'State',
        active: 'active',
        inactive: 'inactive',
        resolution: 'Resolution',
        jpegQuality: 'JPEG Quality',
        payloadSize: 'Payload Size',
        landmarks: 'Landmarks',
        loopDelay: 'Loop Delay',
        canvas: 'Canvas',
        processing: 'Processing',
        busy: 'busy',
        idle: 'idle',
        modelSelected: 'Model Selected',
        modelActive: 'Model Active',
        top1Top2Gap: 'Top1/Top2 Gap',
        requestRtt: 'Request RTT',
        status: 'Status',
        error: 'Error',
        ok: 'ok',
        inference: 'Inference',
        decode: 'Decode',
        preprocess: 'Preprocess',
        processFlow: 'Process Flow',
        total: 'Total',
        hand: 'Hand',
        topCandidate: 'Top Candidate',
        secondCandidate: 'Second Candidate',
        mediapipeAccuracy: 'MediaPipe Accuracy',
        mediapipeLabel: 'MediaPipe Hand Label',
        mediapipeModelComplexity: 'MediaPipe Model Complexity',
        mediapipeMinDetection: 'MediaPipe Min Detection',
        mediapipeStaticMode: 'MediaPipe Static Mode',
        scale: 'Scale',
        shape: 'Shape',
        modelVersionInfo: 'Model Version',
        modelPath: 'Model Path',
        overlayOptions: 'Overlay Options',
        showLandmarks: 'Show Landmarks',
        showV3Features: 'Show v3 Feature Lines',
    },
    de: {
        pageTitle: 'SignAI Live Demo',
        startCamera: 'Kamera starten',
        stopCamera: 'Kamera stoppen',
        fullscreen: 'Vollbild',
        exitFullscreen: 'Vollbild beenden',
        languageTitle: 'Sprache',
        settingsTitle: 'Einstellungen',
        dictionaryTitle: 'Gebärdenwörterbuch',
        closeTitle: 'Schließen',
        modelVersion: 'Modellversion:',
        debugMode: 'Debug-Modus',
        waiting: 'Warten...',
        ready: 'Bereit',
        dictTitle: 'Gebärdensprache Wörterbuch',
        debugCenter: 'Debug-Zentrum',
        roundtrip: 'Roundtrip',
        frames: 'Frames',
        lastUpdate: 'Letztes Update',
        handCutout: 'Handausschnitt',
        cameraStream: 'Kamera / Stream',
        waitingCamera: 'Warte auf Kamera...',
        performance: 'Leistung',
        waitingData: 'Warte auf Daten...',
        system: 'System',
        prediction: 'Vorhersage',
        handMeta: 'Hand-Meta',
        top5Predictions: 'Top-5 Vorhersagen',
        rawResponse: 'Rohantwort (Live)',
        waitingDebugData: 'Warte auf Debug-Daten...',
        noHand: 'Keine Hand',
        cameraActive: 'Kamera aktiv. Starte Vorhersagen...',
        cameraError: 'Fehler: Kein Zugriff auf Kamera. Bitte Berechtigung erteilen.',
        cameraStopped: 'Kamera gestoppt.',
        usingModel: 'Modell aktiv',
        confidence: 'Konfidenz',
        state: 'Status',
        active: 'aktiv',
        inactive: 'inaktiv',
        resolution: 'Auflösung',
        jpegQuality: 'JPEG-Qualität',
        payloadSize: 'Datenmenge',
        landmarks: 'Landmarks',
        loopDelay: 'Loop-Verzögerung',
        canvas: 'Canvas',
        processing: 'Verarbeitung',
        busy: 'beschäftigt',
        idle: 'idle',
        modelSelected: 'Modell gewählt',
        modelActive: 'Modell aktiv',
        top1Top2Gap: 'Top1/Top2 Abstand',
        requestRtt: 'Anfrage RTT',
        status: 'Status',
        error: 'Fehler',
        ok: 'ok',
        inference: 'Inference',
        decode: 'Decode',
        preprocess: 'Vorverarbeitung',
        processFlow: 'Prozessablauf',
        total: 'Gesamt',
        hand: 'Hand',
        topCandidate: 'Top Kandidat',
        secondCandidate: 'Zweiter Kandidat',
        mediapipeAccuracy: 'MediaPipe Genauigkeit',
        mediapipeLabel: 'MediaPipe Hand-Label',
        mediapipeModelComplexity: 'MediaPipe Modellkomplexität',
        mediapipeMinDetection: 'MediaPipe Min. Erkennung',
        mediapipeStaticMode: 'MediaPipe Statischer Modus',
        scale: 'Skalierung',
        shape: 'Form',
        modelVersionInfo: 'Modellversion',
        modelPath: 'Modellpfad',
        overlayOptions: 'Overlay Optionen',
        showLandmarks: 'Landmarks anzeigen',
        showV3Features: 'v3 Feature-Linien anzeigen',
    }
};

function t(key) {
    return translations[currentLang][key] || translations['en'][key] || key;
}

function applyTranslations() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        const text = t(key);
        if (!text) return;
        if (el.id === 'result' && stream) return;
        el.textContent = text;
    });

    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        const key = el.getAttribute('data-i18n-title');
        const text = t(key);
        if (text) el.setAttribute('title', text);
    });

    document.querySelectorAll('[data-i18n-alt]').forEach(el => {
        const key = el.getAttribute('data-i18n-alt');
        const text = t(key);
        if (text) el.setAttribute('alt', text);
    });

    document.title = t('pageTitle');
    document.documentElement.lang = currentLang;
    // Update fullscreen button text
    const fsBtn = document.getElementById('fullscreen-btn');
    if (fsBtn) {
        fsBtn.textContent = isFullscreen ? t('exitFullscreen') : t('fullscreen');
    }
    // Update lang button label
    const langLabel = document.querySelector('.lang-label');
    if (langLabel) langLabel.textContent = currentLang.toUpperCase();
}

function toggleLanguage() {
    currentLang = currentLang === 'en' ? 'de' : 'en';
    const langBtn = document.getElementById('lang-toggle');
    if (langBtn) langBtn.classList.toggle('active', currentLang === 'de');
    applyTranslations();
}

if (toggleV3Features) {
    showV3FeatureOverlay = toggleV3Features.checked;
    toggleV3Features.addEventListener('change', () => {
        showV3FeatureOverlay = toggleV3Features.checked;
    });
}

applyTranslations();
