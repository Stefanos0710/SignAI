# <div align="center">SignAI — Sign Language Translator</div>

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)  
[![License: Non-Commercial](https://img.shields.io/badge/License-Non%20Commercial-red.svg)](LICENSE)  
[![Last Updated](https://img.shields.io/badge/last%20updated-2025--11--24-green.svg)](https://github.com/Stefanos0710/SignAI/commits/main)

</div>

SignAI is an sign language recognition and translation system that uses machine learning to interpret German Sign Language (DGS) (and soon ASL = American Sign Language) in real time and present gloss-style translations. This repository contains code for the recognition engine, frontend UI, and training/inference tooling.

Note: This project is actively developed. v1.0.0 is the first stable major release but still depends on local resources and specific permissions in some environments. See Known Issues & Roadmap for planned improvements.

---

## Note for Moonshot tester
PLease install the version from the main webside https://www.signai.dev and follow the steps below to get started quickly.




## Requirements

- Supported platforms: Windows (soon also for macOS, Linux, Android and IOS)
- Webcam required for live recognition  
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
4. Start the app in dev mode :
   - Option A (Python): python -m signai.app

5. Open the frontend (if applicable) at http://localhost:8000 or as indicated by the startup logs.

Note: Exact run commands may vary depending on your branch or local structure. Check README sections inside subfolders (server/, frontend/) for specific commands.

---

## Models and AI

- Model artifacts: see the `models/` folder. Saved Keras models (checkpoints and final models) and training history JSON files are stored there (e.g. `checkpoint_v28_epoch_01.keras`, `checkpoint_v29_epoch_04.keras`, `history_*.json`).

- Training scripts and workflow:
  - `train.py` and `train-seq2seq.py` implement the main training loops for classification and sequence-to-sequence tasks. They load preprocessed data, build or load a model from `model.py`, configure augmentation, losses, optimizers, and perform training with checkpointing and history logging.
  - Typical training steps: load features -> create dataset -> compile model -> train with callbacks (ModelCheckpoint, CSV/JSON history) -> save final model and history.

- Model structure (high level):
  - Input: sequences of per-frame feature vectors (keypoints or embedding vectors). Shape example: (time_steps, num_features).
  - Temporal encoder: the model processes the time dimension with temporal layers (e.g. stacked 1D conv / Bi-LSTM / Transformer blocks) to capture motion and temporal patterns.
  - Head(s): for classification models a dense + softmax head predicts a single label per sequence; for seq2seq models an encoder-decoder (with attention) predicts token sequences (glosses/translations).
  - The exact architecture is defined in `model.py`. When modifying the architecture, ensure input shape and checkpoint compatibility.

- Metrics & outputs:
  - Training logs record loss and accuracy (or token-level and sequence-level metrics for seq2seq). Use the recorded `history_*.json` files to plot training and validation curves.
  - Evaluation scripts or test notebooks can compute confusion matrices, per-class accuracy, and qualitative inference examples.

- How to run training (quick):
  1. Prepare data (run preprocessing).
  2. Activate virtual environment and install requirements.
  3. Run: `python train.py [--config my_config.yaml]` or `python train-seq2seq.py` depending on task.

- Example image placeholders (replace with real plots):
  - ADD PIC HERE — Model v28 training loss / accuracy plot

  - ADD PIC HERE — Model v29 training loss / accuracy plot

  - ADD PIC HERE — Classification confusion matrix or example inference visualization

---

## Preprocessing of data

This section explains the preprocessing pipeline and the expected input format for training and live inference.

- Key scripts:
  - `preprocessing_train_data.py`: prepares training data from raw videos/frames. Typical steps include frame sampling, keypoint extraction, normalization, sequence length handling (padding/truncation), and saving features in a training-ready format (e.g. .npz or TFRecord).
  - `preprecessing_livedata_web.py` and `api/preprocessing_live_data.py`: lightweight preprocessing for live camera or API inputs. These ensure live inputs match the feature format used during training.
  - `check_dataset.py` and `test_parse_data.py`: validation tools to inspect and verify datasets and feature files.

- Data format and conventions:
  - A sequence is typically a time-ordered array of keypoint vectors per frame: (time_steps, num_features).
  - Coordinate normalization: convert absolute pixel coordinates to a normalized space (relative to person or frame), optionally scale/center and remove outliers.
  - Frame sampling: resample or sample fixed frame rates to create consistent sequence lengths.
  - Padding/truncation: shorter sequences are zero-padded; longer sequences are truncated or sampled to the fixed input length.
  - Labels: classification uses a single label per sequence; seq2seq uses a tokenized target sequence (gloss tokens).

- Recommended workflow:
  1. Collect raw videos/frames under `data/`.
  2. Run `preprocessing_train_data.py` to produce feature files.
  3. Inspect outputs with `check_dataset.py`.
  4. Train using `train.py` or `train-seq2seq.py`.

- Reproducibility tips:
  - Record preprocessing settings (keypoint detector version, frame rate, normalization method) in config files and keep them with corresponding model checkpoints and histories.
  - Version the generated feature files and history outputs together with the model checkpoints for reproducible experiments.

- Image placeholder:
  - ADD PIC HERE — Example of a preprocessed sequence (visualization of keypoints across frames)

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
- General improvements to UI/UX based on user feedback.
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

---


## License

This project is distributed under the license described in the LICENSE file in the repository root. The repo currently uses a non-commercial license badge—please check LICENSE for exact terms and permitted usage.

If you need a different licensing arrangement (e.g., commercial use, enterprise license), please contact us via issues or by the contact information in the repo below.

---

### Media Coverage
- Featured coverage in Süddeutsche Zeitung: (https://www.sueddeutsche.de/muenchen/freising/flughafen-muenchen-jugend-forscht-li.3209469)
- Mentioned in several local newspapers and regional outlets — thanks to local journalists and community supporters for coverage.
---

### Contact

Got questions or need help? Choose the most appropriate channel below.

- General inquiries, partnerships, press, and collaborations  
  [hello@signai.dev](mailto:hello@signai.dev)

- Support, installation issues, and troubleshooting  
  [support@signai.dev](mailto:support@signai.dev)  
  Preferred: please open an issue first at https://github.com/Stefanos0710/SignAI/issues and include steps to reproduce, and any logs or screenshots.

If you need to report a security vulnerability or sensitive data issue, mark the message clearly and we will prioritize acknowledgement.