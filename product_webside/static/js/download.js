// Download page JavaScript - Fetch versions from GitHub and populate dropdowns
const repoUrl = 'https://api.github.com/repos/Stefanos0710/SignAI/releases';

// Platform file mappings
const platformFiles = {
    windows: { extension: '.exe', prefix: 'SignAI-Setup-' },
    macos: { extension: '.dmg', prefix: 'SignAI-' },
    linux: { extension: '.AppImage', prefix: 'SignAI-' },
    android: { extension: '.apk', prefix: 'SignAI-' },
    ios: { extension: '.ipa', prefix: 'SignAI-' }
};

// Fetch releases from GitHub
fetch(repoUrl)
    .then(response => response.json())
    .then(releases => {
        if (!Array.isArray(releases) || releases.length === 0) {
            console.warn("No releases found in repository");
            return;
        }

        // Get the first 5 releases (latest + 4 more)
        const latestReleases = releases.slice(0, 5);

        console.log(`Found ${latestReleases.length} releases`);

        // Populate each platform dropdown
        populateDropdown('windows-version', latestReleases, 'windows');
        populateDropdown('macos-version', latestReleases, 'macos');
        populateDropdown('linux-version', latestReleases, 'linux');
        populateDropdown('android-version', latestReleases, 'android');
        populateDropdown('ios-version', latestReleases, 'ios');
    })
    .catch(error => {
        console.error('Error fetching releases:', error);
    });

// Function to populate dropdown with versions
function populateDropdown(dropdownId, releases, platform) {
    const dropdown = document.getElementById(dropdownId);
    if (!dropdown) return;

    // Clear existing options
    dropdown.innerHTML = '';

    releases.forEach((release, index) => {
        const versionTag = release.tag_name || release.name;
        const fileName = findAssetForPlatform(release, platform);

        if (fileName) {
            const option = document.createElement('option');
            option.value = versionTag;
            option.setAttribute('data-file', fileName);
            option.textContent = index === 0 ? `${versionTag} (Latest)` : versionTag;
            dropdown.appendChild(option);
        }
    });

    // Update download button with first version
    if (dropdown.options.length > 0) {
        updateDownloadButton(dropdownId, dropdown.options[0].getAttribute('data-file'));
    }
}

// Find the correct asset file for a platform from release
function findAssetForPlatform(release, platform) {
    const mapping = platformFiles[platform];
    if (!mapping || !release.assets) return null;

    // Try to find matching asset
    const asset = release.assets.find(a =>
        a.name.includes(mapping.prefix) && a.name.endsWith(mapping.extension)
    );

    if (asset) {
        return asset.browser_download_url;
    }

    // Fallback: construct filename from version tag
    const version = release.tag_name || 'v1.0.0';
    return `${mapping.prefix}${version}${mapping.extension}`;
}

// Update download button href when version changes
function updateDownloadButton(dropdownId, fileUrl) {
    const platformName = dropdownId.replace('-version', '');
    const downloadBox = document.querySelector(`#download-${platformName}`) ||
                       document.querySelector(`.download-${platformName}`);

    if (downloadBox) {
        const downloadBtn = downloadBox.querySelector('.download-btn');
        if (downloadBtn && fileUrl) {
            // If it's a full URL from GitHub, use it directly, otherwise use static path
            if (fileUrl.startsWith('http')) {
                downloadBtn.href = fileUrl;
            } else {
                downloadBtn.href = `/static/downloads/${fileUrl}`;
            }
        }
    }
}

// Version selector change handlers
document.addEventListener('DOMContentLoaded', function() {
    const versionDropdowns = document.querySelectorAll('.version-dropdown');

    versionDropdowns.forEach(dropdown => {
        dropdown.addEventListener('change', function() {
            const selectedOption = this.options[this.selectedIndex];
            const fileName = selectedOption.getAttribute('data-file');
            const platform = this.id.replace('-version', '');

            // Find the corresponding download button
            const downloadBox = this.closest('.download-window, .download-mac, .download-linux, .download-android, .download-ios');
            const downloadBtn = downloadBox?.querySelector('.download-btn');

            if (downloadBtn && fileName) {
                // If it's a full URL from GitHub, use it directly
                if (fileName.startsWith('http')) {
                    downloadBtn.href = fileName;
                } else {
                    downloadBtn.href = `/static/downloads/${fileName}`;
                }

                console.log(`Updated ${platform} download link to: ${fileName}`);
            }
        });
    });

    // Optional: Add download tracking
    const downloadButtons = document.querySelectorAll('.download-btn');
    downloadButtons.forEach(btn => {
        btn.addEventListener('click', function(e) {
            const platform = this.getAttribute('data-platform');
            const downloadBox = this.closest('.download-window, .download-mac, .download-linux, .download-android, .download-ios');
            const version = downloadBox?.querySelector('.version-dropdown')?.value || 'unknown';

            console.log(`Download started: ${platform} - ${version}`);

        });
    });
});
