const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const fpsInput = document.getElementById('fps');
const thresholdInput = document.getElementById('threshold');
const targetCountInput = document.getElementById('target-count');

const statusText = document.getElementById('status-text');
const currentLetter = document.getElementById('current-letter');
const currentProgress = document.getElementById('current-progress');
const timer = document.getElementById('timer');
const totalCount = document.getElementById('total-count');
const letterText = document.getElementById('letter-text');
const refImage = document.getElementById('ref-image');
const result = document.getElementById('result');
const summaryBox = document.getElementById('summary');
const summaryContent = document.getElementById('summary-content');

let stream = null;
let statePollInterval = null;
let frameInterval = null;
let isFrameProcessing = false;
let latestState = null;

function setStatus(message) {
    if (result) result.innerText = message;
}

async function ensureCamera() {
    if (stream) return;
    stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: { ideal: 30 } }
    });
    video.srcObject = stream;
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    video.srcObject = null;
}

function updateRefOverlay(letter) {
    if (!letter) {
        letterText.innerText = 'Zeige Buchstaben: -';
        refImage.removeAttribute('src');
        return;
    }
    letterText.innerText = `Zeige Buchstaben: ${letter}`;
    refImage.src = `/dataset_img/${encodeURIComponent(letter)}`;
}

function renderState(state) {
    latestState = state;
    if (!state) {
        statusText.innerText = 'Warte';
        currentLetter.innerText = '-';
        currentProgress.innerText = '0/0';
        timer.innerText = '0.0s';
        totalCount.innerText = '0';
        updateRefOverlay(null);
        return;
    }

    statusText.innerText = state.status_text;
    currentLetter.innerText = state.current_letter || '-';
    currentProgress.innerText = `${state.current_progress}/${state.target_count}`;
    timer.innerText = `${Number(state.seconds_remaining || 0).toFixed(1)}s`;
    totalCount.innerText = `${state.total_saved}`;
    updateRefOverlay(state.current_letter);

    if (state.phase === 'countdown') {
        setStatus('Warte auf Start der Aufnahme...');
    } else if (state.phase === 'capturing') {
        setStatus('Aufnahme läuft');
    } else if (state.phase === 'pause') {
        setStatus('Buchstabe abgeschlossen');
    } else if (state.phase === 'done') {
        setStatus('Fertig');
        loadSummary();
    } else if (state.phase === 'stopped') {
        setStatus('Gestoppt');
    } else {
        setStatus(state.status_text || 'Warte');
    }
}

async function pollState() {
    try {
        const res = await fetch('/rec_data/state');
        const data = await res.json();
        renderState(data.state || null);
    } catch (error) {
        console.error('State polling failed:', error);
    }
}

async function sendFrame() {
    if (isFrameProcessing) return;
    if (!stream || !latestState || latestState.phase !== 'capturing') return;

    isFrameProcessing = true;
    try {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/jpeg', 0.7);

        const response = await fetch('/rec_data/frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL })
        });

        const payload = await response.json();
        if (payload.state) renderState(payload.state);

        if (payload.accepted === false && payload.reason === 'no_hand') {
            setStatus('Keine Hand erkannt');
        }
    } catch (error) {
        console.error('Frame upload failed:', error);
        setStatus('Fehler beim Senden');
    } finally {
        isFrameProcessing = false;
    }
}

function restartFrameInterval() {
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
    const fps = Math.max(1, Number(fpsInput.value) || 30);
    const intervalMs = Math.max(1, Math.floor(1000 / fps));
    frameInterval = setInterval(sendFrame, intervalMs);
}

async function startRecording() {
    summaryBox.style.display = 'none';
    try {
        await ensureCamera();
    } catch (error) {
        console.error(error);
        setStatus('Kamera verweigert');
        return;
    }

    const payload = {
        fps: Number(fpsInput.value) || 30,
        similarity_threshold: Number(thresholdInput.value) || 0.12,
        target_count: Number(targetCountInput.value) || 100
    };

    try {
        const res = await fetch('/rec_data/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) {
            setStatus(data.error || 'Start fehlgeschlagen');
            return;
        }

        renderState(data.state || null);

        if (statePollInterval) clearInterval(statePollInterval);
        statePollInterval = setInterval(pollState, 250);

        restartFrameInterval();
    } catch (error) {
        console.error(error);
        setStatus('Start fehlgeschlagen');
    }
}

async function stopRecording() {
    try {
        await fetch('/rec_data/stop', { method: 'POST' });
    } catch (error) {
        console.error('stop failed', error);
    }

    if (statePollInterval) {
        clearInterval(statePollInterval);
        statePollInterval = null;
    }
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }

    stopCamera();
    setStatus('Gestoppt');
    pollState();
}

async function loadSummary() {
    try {
        const res = await fetch('/rec_data/summary');
        const data = await res.json();
        const summary = data.summary;
        if (!summary) return;

        const rows = [];
        const perLetter = summary.per_letter || {};
        for (const [letter, count] of Object.entries(perLetter)) {
            rows.push(`<li>${letter}: <strong>${count}</strong></li>`);
        }

        summaryContent.innerHTML = `
            <p>Gesamtanzahl Bilder: <strong>${summary.total_saved}</strong></p>
            <p>Pfad: ${summary.dataset_path}</p>
            <ul>${rows.join('')}</ul>
        `;
        summaryBox.style.display = 'block';
    } catch (error) {
        console.error('summary failed', error);
    }
}

startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);
fpsInput.addEventListener('change', restartFrameInterval);

pollState();
