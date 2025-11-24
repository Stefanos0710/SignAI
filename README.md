# <div align="center">SignAI — Sign Language Translator</div>

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)  
[![License: Non-Commercial](https://img.shields.io/badge/License-Non%20Commercial-red.svg)](LICENSE)  
[![Last Updated](https://img.shields.io/badge/last%20updated-2025--11--24-green.svg)](https://github.com/Stefanos0710/SignAI/commits/main)

</div>

SignAI is an experimental sign language recognition and translation system that uses machine learning to interpret German Sign Language (DGS) (and soon ASL = American Sign Language) in real time and present gloss-style translations. This repository contains code for the recognition engine, frontend UI, and training/inference tooling.

Note: This project is actively developed. v1.0.0 is the first stable major release but still depends on local resources and specific permissions in some environments. See Known Issues & Roadmap for planned improvements.

---

## Table of contents

- [Quick Links](#quick-links)  
- [Highlights (v1.0.0)](#highlights-v100)  
- [Requirements](#requirements)  
- [Installation (End user)](#installation-end-user)  
- [Quick Start (Developer / Local Run)](#quick-start-developer--local-run)  
- [Usage & Tips](#usage--tips)  
- [Technical Details & Model Notes](#technical-details--model-notes)  
- [Known Issues & Workarounds](#known-issues--workarounds)  
- [Roadmap](#roadmap)  
- [Contributing](#contributing)  
- [License](#license)  
- [Support & Reporting Bugs](#support--reporting-bugs)  
- [Acknowledgements & Media](#acknowledgements--media)

---

## Quick Links

- Website / Downloads: https://www.signai.dev/download  
- Issues & support: https://github.com/Stefanos0710/SignAI/issues
- Releases: https://github.com/Stefanos0710/SignAI/releases

---

## Highlights (v1.0.0)

- New DGS model with a vocabulary of over 800 words.  
- Sentence-level translation support for sequences up to 15 words (gloss-style German output).  
- Improved finetuning pipeline and inference performance.  
- Hardened admin-access controls (admin privileges are currently required for some operations; planned to be relaxed).  
- UX improvements for camera handling and clearer first-run behavior.  
- Baseline training metrics (compressed-training setup): Training accuracy ≈ 30%, Validation accuracy ≈ 25%.

---

## Requirements

- Supported platforms: Windows, macOS, Linux (installer available per OS)  
- Webcam (or video input) required for live recognition  
- Disk space: At least 5 GB free (models and caches may need more)  
- Python 3.8+ (for development & building from source)  
- Recommended: GPU (for local training or fast inference), but CPU-only inference is supported

---

## Installation (End user)

1. Go to the downloads page: https://www.signai.dev/download  
2. Download the latest installer for your operating system.  
3. Run the installer and follow the prompts.  
4. After installation, launch the SignAI application.

If the camera feed does not appear on first launch:
- Press the "Switch Camera" button repeatedly until the correct feed appears. Pressing "Switch" cycles available camera devices and usually restores the feed.

First-run behavior:
- On the first run the app must load libraries and model files. This can take several seconds; please wait before expecting the live view to appear.

Security note:
- This release currently requires administrator privileges for some operations (installation, camera permissions, or model management). Future releases will reduce or better encapsulate those requirements.

---

## Quick Start (Developer / Local Run)

The instructions below assume you want to run or develop SignAI locally from source.

1. Clone the repository:
   git clone https://github.com/Stefanos0710/SignAI.git
2. Create and activate a virtual environment:
   python -m venv .venv
   - Windows: .venv\Scripts\activate
   - macOS / Linux: source .venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Start the app in dev mode (example — adjust command to your project layout):
   - Option A (Python): python -m signai.app
   - Option B (Docker): docker-compose up --build
5. Open the frontend (if applicable) at http://localhost:8000 or as indicated by the startup logs.

Note: Exact run commands may vary depending on your branch or local structure. Check README sections inside subfolders (server/, frontend/) for specific commands.

---

## Usage & Tips

- Recording: Click the “Record” button and perform signs. The system attempts to map recognized signs into gloss-style German output.
- Non-professional signers: Recognition quality may vary widely for users who are not trained signers. In some test cases, accuracy for casual or non-professional signing can drop dramatically (reports as low as ~2% in extreme mismatch scenarios).
- Output format: Translations are presented as German glosses (not fully formed grammatical sentences). Expect literal glosses that are intended to be post-processed for natural language rendering.
- If a camera feed is missing: press "Switch Camera" until the correct feed appears. 
- If inference is slow: try closing other camera-using apps and ensure the device has adequate CPU/GPU resources.

---

## Technical details & model notes

- Model: New DGS recognition model (v1.0.0) with >800-word vocabulary and sentence translation up to 15 words.
- Training state for this release:
  - Training accuracy: ~30%
  - Validation accuracy: ~25%
- Training setup: This release was trained with a compressed/limited version of the dataset "PHOENIX-Weather-2014T" due to local compute constraints. Because of this compression and limited resources, the model is a baseline and does not yet represent full production-level accuracy.
- Data & generalization: The current dataset focuses on a subset of vocabulary and is not optimized for open-domain conversational sign language. Expect best results on the supported vocabulary and simpler sentence types (e.g., weather statements or short declarative phrases).
- Future improvements: Planned training on full datasets, improved preprocessing, stronger augmentation, and training on larger compute (supercomputer/cluster) will be used to increase accuracy and vocabulary coverage.

---

## Known issues & workarounds

- Camera feed interference
  - Symptom: No camera image on app open or camera flickers.
  - Cause: Other applications or the operating system may hold camera resources.
  - Workaround: Press "Switch Camera" repeatedly until the feed appears. Close other apps that may use the camera.
- Admin privileges required
  - Symptom: Installer or app requests elevated permissions.
  - Note: This is currently required for some features. Future updates will remove or reduce this requirement.
- First-run delay
  - Symptom: UI or stream is blank for several seconds after first launch.
  - Cause: Libraries and models are being loaded from disk.
  - Workaround: Wait for the initial model load to complete.
- Limited accuracy for casual signers
  - Symptom: Low recognition quality for non-professional signers.
  - Note: This is expected with the current training dataset and will be targeted in future training runs.

---

## Roadmap

Planned focus areas for subsequent major releases:

- Triple the effective accuracy of v1.0.0 by:
  - Training on the full (non-compressed) dataset.
  - Stronger model architectures and finetuning.
  - Faster and more robust preprocessing & data pipelines.
  - Moving training to supercomputers/cloud clusters to enable:
    - Combining multiple datasets
    - Large-scale synthetic data generation & augmentation
- Expand vocabulary coverage (targeting thousands of words over time).
- Reduce admin access requirements and improve security model.
- Improve robustness to camera interference and cross-platform quirks.
- Introduce natural-language rendering (grammatical sentence generation from glosses).

---

## Contributing

We welcome contributions! Below are ways to contribute and a short guide to get started.

How to contribute
1. Star the repository to show support.  
2. Fork the repository and create a feature branch:
   - git checkout -b feat/my-new-feature
3. Implement your changes, include tests where appropriate.
4. Run the test suite and verify formatting/linting (project may include scripts such as `make test` or `scripts/run_tests.sh`).
5. Commit with clear message and push your branch to your fork.
6. Open a Pull Request to the main repository. In the PR description include:
   - What the change is and why it’s needed
   - How to reproduce or test the change
   - Any performance or security considerations
7. Address review feedback and iterate until the PR is ready to merge.

Coding guidelines
- Follow the existing style in the repo (PEP8 for Python).
- Add or update unit/integration tests for functional changes.
- Keep large data or model artifacts out of commits — use model hosting or release assets for large files.

Issue reporting
- Search existing issues before opening a new one.
- Provide reproducible steps, platform/OS info, and logs where relevant.
- Use labels if suggested (bug, enhancement, urgent).

If you are making larger changes (architecture, model training pipeline):
- Open a draft issue or RFC first describing the design and gather feedback from maintainers.

Code of Conduct
- Please follow the project's Code of Conduct (see CODE_OF_CONDUCT.md if available). Be respectful and constructive.

---

## How to customize or extend

- Configuration: Check model and runtime options. Back up your configuration before changes.
- Add models: Place new model files into models/. See docs/ for model packaging format.
- Fine-tuning: Use the training scripts in the training/ folder. Large-scale training requires substantial compute — consider cloud GPUs or cluster resources.
- Frontend changes: The web UI is in frontend/. Edit HTML/CSS/JS there.

If you want to propose a change to the app behavior or default configuration, open a PR with a clear description and, if applicable, a migration guide.

---

## License

This project is distributed under the license described in the LICENSE file in the repository root. The repo currently uses a non-commercial license badge—please check LICENSE for exact terms and permitted usage.

If you need a different licensing arrangement (e.g., commercial use, enterprise license), please contact the maintainers via issues or by the contact information in the repo.

---

## Support & reporting bugs

- Create issues at: https://github.com/Stefanos0710/SignAI/issues  
- Include:
  - OS and version (Windows/macOS/Linux)
  - Application version (v1.0.0)
  - Steps to reproduce
  - Logs (if available) and screenshots/video of the camera feed behavior
  - Hardware info (CPU/GPU, webcam model)

For urgent or large-scale collaboration (dataset sharing, compute access), please open an issue titled "Collaboration / Compute Request" and describe your proposal.

---

## Acknowledgements & Media

### Awards
- 🥈 2nd Place — Jugend forscht 2025

### Media Coverage
- Featured coverage in Süddeutsche Zeitung: (https://www.sueddeutsche.de/muenchen/freising/flughafen-muenchen-jugend-forscht-li.3209469)  
- Mentioned in several local newspapers and regional outlets — thanks to local journalists and community supporters for coverage.

---

## Credits & Thanks

Thanks to all contributors, testers and early adopters. Special thanks to those who helped with data collection, annotation, and testing during the beta phase.

---
