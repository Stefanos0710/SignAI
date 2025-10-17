// Download page JavaScript - Fetch versions from GitHub and populate dropdowns
// Using electron/electron as example repository since it has proper release assets
const repoUrl = 'https://api.github.com/repos/electron/electron/releases';
const repoName = 'electron/electron';

// Platform file mappings - flexibler für verschiedene Dateinamen
const platformFiles = {
    windows: { extensions: ['.exe', '.msi'], keywords: ['win', 'windows', 'setup'] },
    macos: { extensions: ['.dmg', '.pkg', '.zip'], keywords: ['mac', 'darwin'] },
    linux: { extensions: ['.AppImage', '.deb', '.rpm'], keywords: ['linux', 'appimage'] },
    android: { extensions: ['.apk'], keywords: ['android'] },
    ios: { extensions: ['.ipa'], keywords: ['ios'] }
};

// fetch releases from GitHub
fetch(repoUrl)
    .then(response => {
        if (!response.ok) {
            throw new Error(`GitHub API error: ${response.status}`);
        }
        return response.json();
    })
    .then(releases => {
        if (!Array.isArray(releases) || releases.length === 0) {
            console.warn("No releases found in repository");
            return;
        }

        // get the first 10 releases (latest + 9 more)
        const latestReleases = releases.slice(0, 10);

        console.log(`Found ${latestReleases.length} releases`);
        if (latestReleases[0]?.assets) {
            console.log('Available assets in latest release:', latestReleases[0].assets.map(a => a.name));
            console.log('Total assets count:', latestReleases[0].assets.length);
        }

        // Populate each platform dropdown
        populateDropdown('windows-version', latestReleases, 'windows');
        populateDropdown('macos-version', latestReleases, 'macos');
        populateDropdown('linux-version', latestReleases, 'linux');
        populateDropdown('android-version', latestReleases, 'android');
        populateDropdown('ios-version', latestReleases, 'ios');

        // add view all releases links
        addViewAllReleasesLinks();
    })
    .catch(error => {
        console.error('Error fetching releases:', error);
    });

// Function to populate dropdown with versions
function populateDropdown(dropdownId, releases, platform) {
    const dropdown = document.getElementById(dropdownId);
    if (!dropdown) {
        console.warn(`Dropdown ${dropdownId} not found`);
        return;
    }

    // Clear existing options
    dropdown.innerHTML = '';

    let foundCount = 0;
    releases.forEach((release, index) => {
        const versionTag = release.tag_name || release.name;
        const assetUrl = findAssetForPlatform(release, platform);

        if (assetUrl) {
            const option = document.createElement('option');
            option.value = versionTag;
            option.setAttribute('data-file', assetUrl);
            option.textContent = index === 0 ? `${versionTag} (Latest)` : versionTag;
            dropdown.appendChild(option);
            foundCount++;
        }
    });

    // Update download button with first version
    if (dropdown.options.length > 0) {
        updateDownloadButton(dropdownId, dropdown.options[0].getAttribute('data-file'));
        console.log(`✓ Found ${foundCount} versions for ${platform}`);
    } else {
        console.warn(`✗ No compatible assets found for ${platform}`);
        // Add placeholder option
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No releases available';
        dropdown.appendChild(option);
    }

    // Add "View all releases on GitHub" option at the end
    if (foundCount > 0) {
        const divider = document.createElement('option');
        divider.disabled = true;
        divider.textContent = '──────────────────────';
        dropdown.appendChild(divider);

        const viewMoreOption = document.createElement('option');
        viewMoreOption.value = 'view-more';
        viewMoreOption.textContent = '📦 View all releases on GitHub';
        dropdown.appendChild(viewMoreOption);
    }
}

// find the correct asset file for a platform from release
function findAssetForPlatform(release, platform) {
    const mapping = platformFiles[platform];
    if (!mapping || !release.assets || release.assets.length === 0) {
        return null;
    }

    // try to find matching asset with keywords
    const asset = release.assets.find(a => {
        const name = a.name.toLowerCase();
        
        // check if file has correct extension
        const hasValidExtension = mapping.extensions.some(ext => 
            name.endsWith(ext.toLowerCase())
        );
        
        if (!hasValidExtension) return false;
        
        // for better matching, check if name contains platform keywords
        const hasKeyword = mapping.keywords.some(keyword => 
            name.includes(keyword.toLowerCase())
        );
        
        // special case for macOS: prefer .dmg over .zip
        if (platform === 'macos') {
            return hasValidExtension && (name.endsWith('.dmg') || hasKeyword);
        }
        
        return hasValidExtension && (hasKeyword || mapping.extensions.length === 1);
    });

    if (asset) {
        console.log(`✓ Found asset for ${platform}:`, asset.name);
        return asset.browser_download_url;
    }

    // try fallback: just match by extension (first match)
    const fallbackAsset = release.assets.find(a => {
        const name = a.name.toLowerCase();
        return mapping.extensions.some(ext => name.endsWith(ext.toLowerCase()));
    });

    if (fallbackAsset) {
        console.log(`✓ Found fallback asset for ${platform}:`, fallbackAsset.name);
        return fallbackAsset.browser_download_url;
    }

    return null;
}

// update download button href when version changes
function updateDownloadButton(dropdownId, fileUrl) {
    const platformName = dropdownId.replace('-version', '');
    const downloadBox = document.querySelector(`#download-${platformName}`) ||
                       document.querySelector(`.download-${platformName}`);

    if (downloadBox) {
        const downloadBtn = downloadBox.querySelector('.download-btn');
        if (downloadBtn && fileUrl) {
            downloadBtn.href = fileUrl;
            downloadBtn.disabled = false;
            downloadBtn.removeAttribute('download');
            console.log(`✓ Updated ${platformName} button to: ${fileUrl}`);
        } else if (downloadBtn) {
            downloadBtn.href = '#';
            downloadBtn.disabled = true;
        }
    }
}

// add "view all releases" links to the page
function addViewAllReleasesLinks() {
    const releasesContainer = document.getElementById('releases-container');
    if (!releasesContainer) {
        console.warn('Releases container not found');
        return;
    }

    // Clear existing links
    releasesContainer.innerHTML = '';

    // Create a link to view all releases on GitHub
    const viewAllLink = document.createElement('a');
    viewAllLink.href = `https://github.com/${repoName}/releases`;
    viewAllLink.target = '_blank';
    viewAllLink.textContent = 'View all releases';
    viewAllLink.classList.add('view-all-releases');

    releasesContainer.appendChild(viewAllLink);
}

// Version selector change handlers
document.addEventListener('DOMContentLoaded', function() {
    const versionDropdowns = document.querySelectorAll('.version-dropdown');

    versionDropdowns.forEach(dropdown => {
        dropdown.addEventListener('change', function() {
            const selectedOption = this.options[this.selectedIndex];

            // If user clicked "View all releases" option, open GitHub
            if (selectedOption.value === 'view-more') {
                window.open(`https://github.com/${repoName}/releases`, '_blank');
                // Reset dropdown to first option
                this.selectedIndex = 0;
                return;
            }

            const fileName = selectedOption.getAttribute('data-file');

            const downloadBox = this.closest('.download-window, .download-mac, .download-linux, .download-android, .download-ios');
            const downloadBtn = downloadBox?.querySelector('.download-btn');

            if (downloadBtn && fileName) {
                downloadBtn.href = fileName;
                downloadBtn.disabled = false;
                console.log(`Changed download link to: ${fileName}`);
            }
        });
    });

    const downloadButtons = document.querySelectorAll('.download-btn');
    downloadButtons.forEach(btn => {
        btn.addEventListener('click', function(e) {
            // Always get the current selected version from dropdown
            const downloadBox = this.closest('.download-window, .download-mac, .download-linux, .download-android, .download-ios');
            const dropdown = downloadBox?.querySelector('.version-dropdown');

            if (dropdown && dropdown.selectedIndex >= 0) {
                const selectedOption = dropdown.options[dropdown.selectedIndex];
                const fileUrl = selectedOption?.getAttribute('data-file');

                // Update button href with current selection
                if (fileUrl && fileUrl !== '') {
                    this.href = fileUrl;
                    this.disabled = false;
                    console.log(`Download started: ${selectedOption.value} - ${fileUrl}`);
                    // Let the browser handle the download
                    return;
                }
            }

            // If no valid URL, prevent navigation
            if (this.disabled || !this.href || this.href === '#' || this.href.endsWith('#')) {
                e.preventDefault();
                console.warn('No download URL available');
                alert('Please wait for versions to load or select a version.');
                return;
            }
        });
    });
});
