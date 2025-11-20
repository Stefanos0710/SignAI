// Download page JavaScript — simplified: no GitHub API calls, only preprogrammed versions

const repoName = 'Stefanos0710/SignAI';

// Ensure specific manual versions are available in the dropdown and point to known installer URLs
function addManualVersions(dropdownId) {
    const dropdown = document.getElementById(dropdownId);
    if (!dropdown) return;
    const manualTags = ['v1.0.0', 'v0.3.0-beta', 'v0.1.0-alpha'];
    dropdown.innerHTML = '';
    manualTags.forEach((tag, idx) => {
        const option = document.createElement('option');
        option.value = tag;
        // Use the standard GitHub release download URL pattern for SignAI_Setup.exe
        option.setAttribute('data-file', `https://github.com/${repoName}/releases/download/${tag}/SignAI_Setup.exe`);
        option.textContent = idx === 0 ? `${tag} (Latest)` : tag;
        dropdown.appendChild(option);
    });
    // ensure the download button uses the first (latest) option
    if (dropdown.options.length > 0) {
        updateDownloadButton(dropdownId, dropdown.options[0].getAttribute('data-file'));
    }
}

// set download button URL according to dropdown selection
function updateDownloadButton(dropdownId, fileUrl) {
    const platform = dropdownId.replace('-version', '');
    const downloadBox = document.querySelector(`#download-${platform}`) ||
                       document.querySelector(`.download-${platform}`);
    if (downloadBox) {
        const downloadBtn = downloadBox.querySelector('.download-btn');
        if (downloadBtn && fileUrl) {
            downloadBtn.href = fileUrl;
            downloadBtn.disabled = false;
        } else if (downloadBtn) {
            downloadBtn.href = '#';
            downloadBtn.disabled = true;
        }
    }
}

// Handle dropdown changes & update download link
document.addEventListener('DOMContentLoaded', function() {
    addManualVersions('windows-version');

    document.querySelectorAll('.version-dropdown').forEach(dropdown => {
        dropdown.addEventListener('change', function() {
            const selectedOption = this.options[this.selectedIndex];
            if (selectedOption.value === 'view-more') {
                window.open(`https://github.com/${repoName}/releases`, '_blank');
                this.selectedIndex = 0;
                return;
            }
            const fileName = selectedOption.getAttribute('data-file');
            const downloadBox = this.closest('.download-window, .download-mac, .download-linux, .download-android, .download-ios');
            const downloadBtn = downloadBox?.querySelector('.download-btn');
            if (downloadBtn && fileName) {
                downloadBtn.href = fileName;
                downloadBtn.disabled = false;
            }
        });
    });

    document.querySelectorAll('.download-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            const downloadBox = this.closest('.download-window, .download-mac, .download-linux, .download-android, .download-ios');
            const dropdown = downloadBox?.querySelector('.version-dropdown');
            if (dropdown && dropdown.selectedIndex >= 0) {
                const selectedOption = dropdown.options[dropdown.selectedIndex];
                const fileUrl = selectedOption?.getAttribute('data-file');
                if (fileUrl && fileUrl !== '') {
                    this.href = fileUrl;
                    this.disabled = false;
                    return;
                }
            }
            if (this.disabled || !this.href || this.href === '#' || this.href.endsWith('#')) {
                e.preventDefault();
                alert('Please select a version to download!');
            }
        });
    });
});
