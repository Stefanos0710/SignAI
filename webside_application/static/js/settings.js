class SettingsManager {
    constructor() {
        this.settings = {
            theme: localStorage.getItem('theme') || 'light',
            highContrast: localStorage.getItem('highContrast') === 'true',
            cameraQuality: localStorage.getItem('cameraQuality') || 'medium',
            mirrorCamera: localStorage.getItem('mirrorCamera') !== 'false',
            autoFocus: localStorage.getItem('autoFocus') !== 'false'
        };

        this.elements = {
            panel: document.getElementById('settings-panel'),
            backdrop: document.querySelector('.settings-backdrop'),
            openButton: document.getElementById('settings-btn'),
            closeButton: document.getElementById('close-settings'),
            saveButton: document.getElementById('save-settings'),
            resetButton: document.getElementById('reset-settings'),
            themeButtons: document.querySelectorAll('.theme-btn'),
            contrastToggle: document.getElementById('contrast-toggle'),
            qualitySelect: document.getElementById('camera-quality'),
            mirrorToggle: document.getElementById('mirror-toggle'),
            autofocusToggle: document.getElementById('autofocus-toggle')
        };

        this.defaultSettings = {
            theme: 'light',
            highContrast: false,
            cameraQuality: 'medium',
            mirrorCamera: true,
            autoFocus: true
        };

        this.initialize();
    }

    initialize() {
        this.loadSettings();
        this.attachEventListeners();
        this.applySettings();
    }

    loadSettings() {
        // Load settings from localStorage
        Object.keys(this.settings).forEach(key => {
            const savedValue = localStorage.getItem(key);
            if (savedValue !== null) {
                this.settings[key] = 
                    savedValue === 'true' ? true :
                    savedValue === 'false' ? false :
                    savedValue;
            }
        });

        // Update UI to reflect current settings
        this.updateUI();
    }

    updateUI() {
        // Update theme buttons
        this.elements.themeButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.theme === this.settings.theme);
        });

        // Update toggles and selects
        if (this.elements.contrastToggle) {
            this.elements.contrastToggle.checked = this.settings.highContrast;
        }
        if (this.elements.qualitySelect) {
            this.elements.qualitySelect.value = this.settings.cameraQuality;
        }
        if (this.elements.mirrorToggle) {
            this.elements.mirrorToggle.checked = this.settings.mirrorCamera;
        }
        if (this.elements.autofocusToggle) {
            this.elements.autofocusToggle.checked = this.settings.autoFocus;
        }
    }

    attachEventListeners() {
        // Panel open/close
        this.elements.openButton.addEventListener('click', () => this.openPanel());
        this.elements.closeButton.addEventListener('click', () => this.closePanel());
        this.elements.backdrop.addEventListener('click', () => this.closePanel());

        // Theme switching
        this.elements.themeButtons.forEach(btn => {
            btn.addEventListener('click', () => this.handleThemeChange(btn.dataset.theme));
        });

        // Other settings
        if (this.elements.contrastToggle) {
            this.elements.contrastToggle.addEventListener('change', (e) => {
                this.settings.highContrast = e.target.checked;
                this.applySettings();
            });
        }

        if (this.elements.qualitySelect) {
            this.elements.qualitySelect.addEventListener('change', (e) => {
                this.settings.cameraQuality = e.target.value;
                this.applySettings();
            });
        }

        if (this.elements.mirrorToggle) {
            this.elements.mirrorToggle.addEventListener('change', (e) => {
                this.settings.mirrorCamera = e.target.checked;
                this.applySettings();
            });
        }

        if (this.elements.autofocusToggle) {
            this.elements.autofocusToggle.addEventListener('change', (e) => {
                this.settings.autoFocus = e.target.checked;
                this.applySettings();
            });
        }

        // Save and Reset buttons
        this.elements.saveButton.addEventListener('click', () => this.saveSettings());
        this.elements.resetButton.addEventListener('click', () => this.resetSettings());

        // Handle escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isPanelOpen()) {
                this.closePanel();
            }
        });
    }

    handleThemeChange(theme) {
        this.settings.theme = theme;
        this.elements.themeButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.theme === theme);
        });
        this.applySettings();
    }

    applySettings() {
        // Apply theme
        document.documentElement.setAttribute('data-theme', this.settings.theme);
        
        // Apply high contrast
        document.documentElement.classList.toggle('high-contrast', this.settings.highContrast);

        // Apply camera settings
        const video = document.getElementById('video');
        if (video) {
            video.style.transform = this.settings.mirrorCamera ? 'scaleX(-1)' : 'none';
        }

        // Dispatch event for other components
        const event = new CustomEvent('settingsChanged', { detail: this.settings });
        document.dispatchEvent(event);
    }

    saveSettings() {
        // Save to localStorage
        Object.entries(this.settings).forEach(([key, value]) => {
            localStorage.setItem(key, value);
        });

        // Show success notification
        this.showNotification('Settings saved successfully!');
        this.closePanel();
    }

    resetSettings() {
        // Reset to defaults
        this.settings = { ...this.defaultSettings };
        
        // Clear localStorage
        Object.keys(this.settings).forEach(key => {
            localStorage.removeItem(key);
        });

        // Update UI and apply settings
        this.updateUI();
        this.applySettings();
        
        // Show notification
        this.showNotification('Settings reset to defaults');
    }

    showNotification(message, isError = false) {
        const notification = document.createElement('div');
        notification.className = `notification ${isError ? 'error' : 'success'}`;
        notification.textContent = message;
        document.body.appendChild(notification);

        // Remove notification after 3 seconds
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    openPanel() {
        this.elements.panel.classList.add('active');
        this.elements.backdrop.classList.add('active');
    }

    closePanel() {
        this.elements.panel.classList.remove('active');
        this.elements.backdrop.classList.remove('active');
    }

    isPanelOpen() {
        return this.elements.panel.classList.contains('active');
    }
}

// Initialize Settings Manager when document is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.settingsManager = new SettingsManager();
});