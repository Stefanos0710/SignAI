class ProcessingManager {
    constructor() {
        this.stepTimes = {
            keypoints: { start: 0, end: 0 },
            preprocessing: { start: 0, end: 0 },
            inference: { start: 0, end: 0 }
        };
        this.currentStep = null;
        this.processingStartTime = 0;

        this.initializeElements();
        this.setupSocketListeners();
        this.startTimeUpdate();
    }

    initializeElements() {
        this.processingSection = document.getElementById('processing-section');
        this.statusText = document.querySelector('.status-text');
        this.progressFill = document.querySelector('.progress-fill');
        this.resultContainer = document.querySelector('.result-container');
        this.circleLoader = document.querySelector('.circle-loader');
        this.currentTimeElement = document.getElementById('current-time');
    }

    startTimeUpdate() {
        setInterval(() => {
            const now = new Date();
            const timeString = now.toISOString().replace('T', ' ').split('.')[0] + ' UTC';
            this.currentTimeElement.textContent = timeString;
        }, 1000);
    }

    updateStepTime(step) {
        const stepElement = document.querySelector(`[data-step="${step}"]`);
        if (!stepElement) return;

        const timeElement = stepElement.querySelector('.step-time');
        if (!timeElement) return;

        const times = this.stepTimes[step];
        if (times.end) {
            const duration = ((times.end - times.start) / 1000).toFixed(2);
            timeElement.textContent = `${duration}s`;
        }
    }

    startStep(step) {
        this.currentStep = step;
        this.stepTimes[step].start = Date.now();
        if (!this.processingStartTime) {
            this.processingStartTime = Date.now();
        }

        const stepElement = document.querySelector(`[data-step="${step}"]`);
        if (stepElement) {
            stepElement.classList.add('active');
            stepElement.querySelector('.step-status').textContent = '⏳';
        }
    }

    completeStep(step) {
        this.stepTimes[step].end = Date.now();
        this.updateStepTime(step);

        const stepElement = document.querySelector(`[data-step="${step}"]`);
        if (stepElement) {
            stepElement.classList.remove('active');
            stepElement.classList.add('completed');
            stepElement.querySelector('.step-status').textContent = '✓';
        }
    }

    setupSocketListeners() {
        const socket = io();

        socket.on('processing_update', (data) => {
            if (!this.processingSection.style.display ||
                this.processingSection.style.display === 'none') {
                this.processingSection.style.display = 'block';
            }

            switch(data.phase) {
                case 'keypoints':
                    if (data.status === 'started') this.startStep('keypoints');
                    if (data.status === 'completed') this.completeStep('keypoints');
                    this.updateProgress(30);
                    break;

                case 'preprocessing':
                    if (data.status === 'started') this.startStep('preprocessing');
                    if (data.status === 'completed') this.completeStep('preprocessing');
                    this.updateProgress(60);
                    break;

                case 'inference':
                    if (data.status === 'started') this.startStep('inference');
                    if (data.status === 'completed') {
                        this.completeStep('inference');
                        this.showResult(data.result);
                    }
                    this.updateProgress(100);
                    break;

                case 'error':
                    this.handleError(data.message);
                    break;
            }
        });
    }

    updateProgress(percent) {
        this.progressFill.style.width = `${percent}%`;
    }

    showResult(result) {
        const totalTime = ((Date.now() - this.processingStartTime) / 1000).toFixed(2);

        this.circleLoader.classList.add('load-complete');
        this.resultContainer.style.display = 'block';
        this.resultContainer.querySelector('.result-text').textContent = result;
        this.resultContainer.querySelector('.total-time').textContent =
            `Total Processing Time: ${totalTime}s`;
        this.resultContainer.querySelector('.completion-time').textContent =
            `Completed at: ${new Date().toISOString().replace('T', ' ').split('.')[0]} UTC`;

        // Show timing breakdown
        const timingBreakdown = document.createElement('div');
        timingBreakdown.className = 'timing-breakdown';
        timingBreakdown.innerHTML = Object.entries(this.stepTimes)
            .map(([step, times]) => {
                const duration = ((times.end - times.start) / 1000).toFixed(2);
                return `<div class="timing-item">
                    <span class="timing-label">${step}:</span>
                    <span class="timing-value">${duration}s</span>
                </div>`;
            })
            .join('');

        this.resultContainer.appendChild(timingBreakdown);
    }

    handleError(message) {
        const stepElement = document.querySelector(`[data-step="${this.currentStep}"]`);
        if (stepElement) {
            stepElement.classList.add('error');
            stepElement.querySelector('.step-status').textContent = '❌';
        }

        this.statusText.textContent = 'Error occurred';
        this.statusText.classList.add('error');
    }

    reset() {
        this.stepTimes = {
            keypoints: { start: 0, end: 0 },
            preprocessing: { start: 0, end: 0 },
            inference: { start: 0, end: 0 }
        };
        this.currentStep = null;
        this.processingStartTime = 0;

        // Reset UI elements
        document.querySelectorAll('.step-item').forEach(step => {
            step.className = 'step-item';
            step.querySelector('.step-status').textContent = '';
            step.querySelector('.step-time').textContent = '0.00s';
        });

        this.progressFill.style.width = '0%';
        this.circleLoader.classList.remove('load-complete');
        this.resultContainer.style.display = 'none';
        this.statusText.textContent = 'Initializing...';
        this.statusText.classList.remove('error');
    }
}

// Initialize Processing Manager
document.addEventListener('DOMContentLoaded', () => {
    window.processingManager = new ProcessingManager();
});
