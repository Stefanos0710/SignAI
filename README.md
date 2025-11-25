<div align="center">

# SignAI — Sign Language Translator

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)  
[![License: Non-Commercial](https://img.shields.io/badge/License-Non%20Commercial-red.svg)](LICENSE)  
[![Last Updated](https://img.shields.io/badge/last%20updated-2025--11--24-green.svg)](https://github.com/Stefanos0710/SignAI/commits/main)

</div>

SignAI is an experimental sign language recognition and translation system that uses machine learning to interpret German Sign Language (DGS) in real time and produce gloss-style translations. This repository contains the recognition engine, frontend UI, preprocessing & training pipelines, and inference tooling.

Primary languages: Python (core), CSS/HTML/JavaScript (frontend).

> Note: v1.0.0 is the first stable major release. The project is actively developed and some operational aspects (admin privileges, resource requirements) are still being refined. See Known Issues & Roadmap for details.

---

## Table of contents

- [Quick links](#quick-links)  
- [Highlights (v1.0.0)](#highlights-v100)  
- [Requirements](#requirements)  
- [Installation (end user)](#installation-end-user)  
- [Quick start (developer / local run)](#quick-start-developer--local-run)  
- [Models & AI](#models--ai)  
  - [Model artifacts and training workflow](#model-artifacts-and-training-workflow)  
  - [Seq2Seq architecture (detailed)](#seq2seq-architecture-detailed)  
  - [Training visualizations](#training-visualizations)  
- [Preprocessing](#preprocessing)  
- [Usage & tips](#usage--tips)  
- [Technical notes & baseline metrics](#technical-notes--baseline-metrics)  
- [Known issues & workarounds](#known-issues--workarounds)  
- [Roadmap](#roadmap)  
- [Contributing](#contributing)  
- [License](#license)  
- [Media & acknowledgements](#media--acknowledgements)  
- [Contact](#contact)

---

## Quick links

- Website / Downloads: https://www.signai.dev/download  
- Issues & support: https://github.com/Stefanos0710/SignAI/issues  
- Releases & changelog: https://github.com/Stefanos0710/SignAI/releases  
- Full repository: https://github.com/Stefanos0710/SignAI

---

## Highlights (v1.0.0)

- New DGS model with a vocabulary of 800+ gloss tokens.  
- Sentence-level gloss translation for sequences up to 15 tokens.  
- Improved finetuning and a faster, more secure inference pipeline.  
- Operational and UX improvements for camera handling and startup.  
- Baseline training metrics (compressed dataset): training accuracy ≈ 30%, validation ≈ 25%.

---

## Note for Moonshot tester

Please install the latest release from https://www.signai.dev and follow the Quick Start steps below for a fast setup.

---

## Requirements

- Supported platforms: Windows (primary). macOS, Linux, Android and iOS builds planned.  
- Webcam or compatible video input for live recognition.  
- Disk space: minimum 5 GB free (models/caches may require more).  
- Python 3.8+ (for development and source builds).  
- Recommended: GPU for faster inference and training; CPU-only inference is supported but slower.

---

## Installation (end user)

1. Visit https://www.signai.dev/download and download the appropriate installer for your OS.  
2. Run the installer and follow the on-screen instructions.  
3. Launch the SignAI application.

Troubleshooting
- If the camera feed does not appear on startup, click the “Switch Camera” button repeatedly until the correct feed appears (the OS or other apps might lock the camera).  
- First run may take several seconds while libraries and model files load; please wait for the UI to become responsive.

Security note
- Some operations in this release may require administrator privileges (installation, camera access, certain model management tasks). Future releases will reduce these requirements or provide safer alternatives.

---

## Quick start (developer / local run)

1. Clone the repository:
   git clone https://github.com/Stefanos0710/SignAI.git
2. Create & activate a virtual environment:
   python -m venv .venv
   - Windows: .venv\Scripts\activate
   - macOS / Linux: source .venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Start the app (development mode):
   cd app
   python app.py

For training:
- Use `python train.py` for single-word classification or `python train-seq2seq.py` for sentence-level training (see Models & AI).

---

## Models & AI

### Model artifacts and training workflow

- Model artifacts are stored in `models/` (Keras checkpoints, final models, and training history JSON/CSV files).  
- Training scripts:
  - `train.py` — single-word classification training loop.  
  - `train-seq2seq.py` — sequence-to-sequence training for sentence-level gloss translation.
- Typical training flow:
  1. Run preprocessing to produce feature files (keypoint embeddings or frame features).  
  2. Create TF/PyTorch datasets and dataloaders.  
  3. Build or load a model from `model.py`.  
  4. Configure augmentation, optimizers, and losses.  
  5. Train with callbacks (ModelCheckpoint, EarlyStopping, CSV/JSON history).  
  6. Save final model and training history.

When changing model architecture, keep checkpoint compatibility in mind (naming conventions or conversion scripts help migration).

### Seq2Seq architecture (detailed)

This project’s sentence translation uses an encoder–decoder (seq2seq) architecture with additive attention. Summary of the implemented architecture:

- Encoder
  - Input: variable-length sequences of per-frame features (shape: batch × time_steps × num_features).  
  - Masking to ignore padded frames.  
  - Bidirectional LSTM (returning sequences and forward/backward final states).  
  - Concatenate forward and backward states to initialize the decoder.
- Decoder
  - Token input sequence (previous tokens during training — teacher forcing).  
  - Embedding layer (mask_zero=True).  
  - LSTM initialized with concatenated encoder states.
- Attention
  - Additive (Bahdanau-style) attention between decoder outputs and encoder outputs to compute a time-dependent context vector.
- Output
  - Concatenate decoder output and attention context.  
  - Dense softmax projection to produce token probabilities over the gloss vocabulary.

Design rationale and training notes are documented in MODEL_ARCHITECTURE.md and the code comments in `model.py`.

### Training visualizations

Below are example training history plots and diagnostics (replace with updated figures from `models/` if available):

- Training history (example run — mode 28):  
<img width="1200" height="400" alt="training_history_v28" src="https://github.com/user-attachments/assets/801ccbc6-f84f-4d26-840a-45ad8466db8d" />

- Training history (example run — model 29):  
<img width="1200" height="400" alt="training_history_v29" src="https://github.com/user-attachments/assets/130e6ee7-dbac-455e-bbeb-dffee992ed28" />

- Classification training snapshot:  
<img width="1200" height="400" alt="training_20251122_103141" src="https://github.com/user-attachments/assets/3913ef63-bfa5-4cb6-a859-2445e2a7761d" />

---

## Preprocessing

- Key scripts:
  - `preprocessing_train_data.py` — prepares training features from raw videos/frames (frame sampling, keypoint extraction, normalization, padding/truncation).  
  - `preprecessing_livedata_web.py` / `api/preprocessing_live_data.py` — lightweight live preprocessing pipeline for camera / API inputs.  
- Data format
  - A sequence is a time-ordered array of per-frame feature vectors: (time_steps, num_features).  
  - Coordinate normalization is recommended (relative to person or frame) to reduce variation.  
  - Short sequences are zero-padded; long sequences are truncated or sampled to a fixed maximum length.
- Recommended workflow
  1. Collect raw videos under `data/`.  
  2. Run `preprocessing_train_data.py` to generate feature files.  
  3. Inspect features with `check_dataset.py`.  
  4. Train with `train.py` or `train-seq2seq.py`.

Example visualization (keypoint & pose preprocessing example using MediaPipe / Holistic output):  
<img width="850" height="958" alt="MediaPipe-Holistic-API" src="https://github.com/user-attachments/assets/1f0fa089-ae88-4ec1-8423-557f37a89cd5" />

---

## Usage & tips

- Recording: Press “Record” and perform signs. The output is gloss-style German tokens — not fully grammatical sentences.  
- Non-professional signers: Expect variable recognition quality. Casual or atypical signing can drop accuracy substantially.  
- Camera feed missing: Press "Switch Camera" until the correct feed appears. Close other apps that may hold the webcam.  
- Slow inference: Close other camera-using apps, free CPU/GPU resources, or use a device with a GPU.

---

## Technical notes & baseline metrics

- Model: DGS recognition model v1.0.0 with >800 gloss tokens and sentence translation up to 15 tokens.  
- Dataset (training baseline): compressed subsets of PHOENIX-Weather-2014T due to local compute limits — this explains lower initial accuracy.  
- Baseline metrics:
  - Training accuracy: ~30%  
  - Validation accuracy: ~25%
- These metrics are a starting point; retraining on full datasets, improved preprocessing and larger models are planned.

---

## Known issues & workarounds

- Camera feed interference
  - Symptom: No camera image or flicker.  
  - Workaround: Press "Switch Camera" repeatedly; close other apps using the camera.
- Admin privileges required
  - Symptom: Installer or app requests elevated permissions.  
  - Note: This release may need admin access for certain tasks. Reductions to this requirement are planned.
- First-run delay
  - Symptom: Blank UI or delayed stream on first launch.  
  - Cause: Libraries and models are loading from disk.  
  - Workaround: Wait a few seconds for the initial load.
- Limited accuracy for casual signers
  - Symptom: Low recognition quality for non-professional or out-of-distribution signers.  
  - Note: Addressed in future training/augmentation plans.

---

## Roadmap

Planned next steps and goals:

- Improve accuracy substantially (target: 3x improvement over v1.0.0) by:
  - Training on full (non-compressed) datasets.  
  - Moving training to larger compute (cloud / supercomputers).  
  - Combining multiple datasets and adding synthetic augmentation.  
  - Exploring transformer-based architectures and stronger preprocessing.
- Expand vocabulary coverage (thousands of gloss tokens over time).  
- Reduce admin-access requirements and harden camera handling.  
- Add natural language rendering (convert glosses to grammatical sentences) and multilingual support (ASL planned).

---

## Contributing

We welcome contributions:

1. Star the repo.  
2. Fork and create a branch: `git checkout -b feat/my-change`  
3. Add tests and documentation for changes.  
4. Run the test suite and linters.  
5. Open a Pull Request with a clear description, test instructions and any migration notes.

Please avoid committing large model binaries — use release assets or external model hosting.

---

## License

See the LICENSE file in the repository root. The project currently uses a non-commercial license; contact the maintainers if you require a different arrangement.

---

## Media & acknowledgements

- 🥈 2nd Place — Jugend forscht 2025  
- Featured coverage in Süddeutsche Zeitung and several local/regional outlets (links in repository).

---

## Contact

- General / partnerships / press: [hello@signai.dev](mailto:hello@signai.dev)  
- Support / troubleshooting: [support@signai.dev](mailto:support@signai.dev) — preferred: open an issue first at https://github.com/Stefanos0710/SignAI/issues with reproduction steps and logs.

If you report a security vulnerability, mark the message “SECURITY” and include reproduction steps; security reports are prioritized.

---
