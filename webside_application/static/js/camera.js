document.addEventListener('DOMContentLoaded', () => {
    // MediaRecorder instance and state
    let mediaRecorder;
    let recordedChunks = [];
    let isRecording = false;
    let stream;

    // Create countdown overlay
    const countdownOverlay = document.createElement('div');
    countdownOverlay.className = 'countdown-overlay';
    document.body.appendChild(countdownOverlay);

    // Add styles for animations and UI elements
    const style = document.createElement('style');
    style.textContent = `
        .countdown-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            display: none;
            justify-content: center;
            align-items: center;
            font-size: 120px;
            color: white;
            z-index: 1000;
        }

        .countdown-number {
            animation: pulse 1s ease-in-out;
            text-shadow: 0 0 20px rgba(52, 152, 219, 0.8);
        }

        .recording-indicator {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: red;
            animation: blink 1s infinite;
        }

        .button-recording {
            background-color: #e74c3c !important;
            animation: pulseButton 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(3); opacity: 0; }
            50% { transform: scale(1.5); opacity: 0.7; }
            100% { transform: scale(1); opacity: 1; }
        }

        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }

        @keyframes pulseButton {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        @keyframes slideIn {
            from { transform: translateX(-50%) translateY(-100%); opacity: 0; }
            to { transform: translateX(-50%) translateY(0); opacity: 1; }
        }

        @keyframes slideOut {
            from { transform: translateX(-50%) translateY(0); opacity: 1; }
            to { transform: translateX(-50%) translateY(-100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);

    // Get video element
    const video = document.getElementById('video');
    const startRecordingButton = document.getElementById('start-recording');
    const switchCameraButton = document.getElementById('switch-camera');

    // Camera configuration based on settings
    function getCameraConfig() {
        const qualitySelect = document.getElementById('camera-quality');
        const quality = qualitySelect ? qualitySelect.value : 'medium';
        const configs = {
            low: { width: 640, height: 480, frameRate: 24 },
            medium: { width: 1280, height: 720, frameRate: 30 },
            high: { width: 1920, height: 1080, frameRate: 30 }
        };
        return {
            video: {
                ...configs[quality],
                facingMode: 'user'
            },
            audio: false
        };
    }

    // Initialize camera with settings
    async function initCamera() {
        try {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }

            stream = await navigator.mediaDevices.getUserMedia(getCameraConfig());
            video.srcObject = stream;
            await video.play();

            // Apply mirror effect based on settings
            const mirrorToggle = document.getElementById('mirror-toggle');
            if (mirrorToggle) {
                video.style.transform = mirrorToggle.checked ? 'scaleX(-1)' : 'none';
            }

            // Setup MediaRecorder with optimal settings
            const options = {
                mimeType: 'video/webm;codecs=vp8',
                videoBitsPerSecond: 2500000 // 2.5 Mbps
            };

            try {
                mediaRecorder = new MediaRecorder(stream, options);
            } catch (e) {
                console.error('MediaRecorder error:', e);
                mediaRecorder = new MediaRecorder(stream);
            }

            setupMediaRecorderEvents();
            startRecordingButton.disabled = false;
            switchCameraButton.disabled = false;

        } catch (err) {
            console.error('Error accessing camera:', err);
            showNotification('Camera access denied or not available!', true);
        }
    }

        // Setup MediaRecorder event handlers
    function setupMediaRecorderEvents() {
        mediaRecorder.ondataavailable = function(event) {
            if (event.data && event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async function() {
            if (recordedChunks.length === 0) {
                showNotification('No video data recorded!', true);
                return;
            }

            const blob = new Blob(recordedChunks, {
                type: 'video/webm'
            });

            try {
                // Zuerst den Ordner leeren
                await fetch('/clear-folder', { method: 'POST' });

                // Video an den Server senden
                const formData = new FormData();
                formData.append('video', blob, 'recorded_video.webm');

                const response = await fetch('/save-video', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                if (data.success) {
                    showNotification('Video uploaded successfully!');
                    document.getElementById('processing-section').style.display = 'block';
                } else {
                    throw new Error(data.error || 'Failed to save video');
                }
            } catch (error) {
                console.error('Error saving video:', error);
                showNotification('Error saving video: ' + error.message, true);
            }

            recordedChunks = [];
        };
    }

    // Notification system
    function showNotification(message, isError = false) {
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 15px 30px;
            background: ${isError ? '#e74c3c' : '#2ecc71'};
            color: white;
            border-radius: 5px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            z-index: 1001;
            animation: slideIn 0.5s ease-out;
        `;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOut 0.5s ease-in';
            setTimeout(() => notification.remove(), 500);
        }, 3000);
    }

    // Countdown function
    function startCountdown() {
        return new Promise((resolve) => {
            let count = 3;
            countdownOverlay.style.display = 'flex';

            const updateCountdown = () => {
                if (count > 0) {
                    countdownOverlay.innerHTML = `
                        <div class="countdown-number">${count}</div>
                    `;
                    count--;
                    setTimeout(updateCountdown, 1000);
                } else {
                    countdownOverlay.style.display = 'none';
                    resolve();
                }
            };

            updateCountdown();
        });
    }

    // Recording indicator
    function createRecordingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'recording-indicator';
        document.getElementById('video-section').appendChild(indicator);
        return indicator;
    }

    // Helper function to stop recording
    function stopRecording(button) {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        isRecording = false;

        // Update UI
        button.innerHTML = `
            <span class="button-icon">🎥</span>
            Start Recording
        `;
        button.classList.remove('button-recording');
        const indicator = document.querySelector('.recording-indicator');
        if (indicator) indicator.remove();
    }

    // Switch camera
    let frontCamera = true;
    async function switchCamera() {
        frontCamera = !frontCamera;
        const constraints = {
            video: {
                facingMode: frontCamera ? 'user' : 'environment'
            },
            audio: false
        };

        try {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            setupMediaRecorderEvents();
        } catch (err) {
            console.error('Error switching camera:', err);
            showNotification('Error switching camera. Your device might not have multiple cameras.', true);
        }
    }

    // Initialize recording controls
    function initRecordingControls() {
        startRecordingButton.addEventListener('click', async function() {
            if (!isRecording) {
                try {
                    // Start countdown
                    this.disabled = true;
                    await startCountdown();
                    this.disabled = false;

                    // Start recording
                    recordedChunks = [];
                    mediaRecorder.start(1000); // Get data every second
                    isRecording = true;

                    // Update UI
                    this.innerHTML = `
                        <span class="button-icon">⏹️</span>
                        Stop Recording
                    `;
                    this.classList.add('button-recording');
                    createRecordingIndicator();

                    showNotification('Recording started!');

                    // Safety timeout - stop after 30 seconds
                    setTimeout(() => {
                        if (isRecording) {
                            stopRecording(this);
                        }
                    }, 30000);

                } catch (error) {
                    console.error('Error starting recording:', error);
                    showNotification('Error starting recording: ' + error.message, true);
                    this.disabled = false;
                }
            } else {
                stopRecording(this);
            }
        });

        // Switch camera button event
        switchCameraButton.addEventListener('click', switchCamera);
    }

    // Handle quality changes
    const qualitySelect = document.getElementById('camera-quality');
    if (qualitySelect) {
        qualitySelect.addEventListener('change', initCamera);
    }

    // Handle mirror mode toggle
    const mirrorToggle = document.getElementById('mirror-toggle');
    if (mirrorToggle) {
        mirrorToggle.addEventListener('change', () => {
            video.style.transform = mirrorToggle.checked ? 'scaleX(-1)' : 'none';
        });
    }

    // Initialize camera when page loads
    initCamera();
    initRecordingControls();
});
