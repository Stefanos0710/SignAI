<div align="center">

<img src="app/icons/icon.png" alt="SignAI" width="130"/>

# SignAI — Sign Language Translator

**Real-time Sign Language recognition and gloss translation using deep learning**

<br>


[![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10%E2%80%933.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras 3.7](https://img.shields.io/badge/Keras-3.7-red?logo=keras&logoColor=white)](https://keras.io/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-brightgreen)](https://mediapipe.dev)
[![PySide6](https://img.shields.io/badge/PySide6-6.5%E2%80%936.7-blue?logo=qt)](https://wiki.qt.io/Qt_for_Python)
[![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)](https://flask.palletsprojects.com/)



[![License](https://img.shields.io/badge/License-Non%20Commercial-red)](/LICENSE)
[![Website](https://img.shields.io/badge/Website-signai.dev-brightgreen)](https://signai.dev)
[![Hosting](https://img.shields.io/badge/Hosting-Vercel-black?logo=vercel)](https://vercel.com)
[![Media](https://img.shields.io/badge/In%20the%20news-SZ%20%7C%20BR%20%7C%20Jugend%20forscht-blue)](https://www.sueddeutsche.de/muenchen/landkreismuenchen/ismaning-gymnasium-jugend-forscht-ki-gebaerdensprache-li.3475266)
[![Hackatime](https://hackatime-badge.hackclub.com/U090BP84F7F/SignAI)](https://hackatime-badge.hackclub.com/U090BP84F7F/SignAI)

</div>

SignAI is a real-time sign language recognition and translation system for German Sign Language (DGS). It uses a sequence-to-sequence model with multi-head attention, trained on MediaPipe Holistic keypoint features. The project won 1st place at the Jugend forscht state competition and received coverage in SZ, BR, and other media outlets.

Primary languages: Python (core, app), CSS/HTML/JavaScript (product website).

---

## Table of Contents

- [SignAI — Sign Language Translator](#signai--sign-language-translator)
  - [Table of Contents](#table-of-contents)
  - [How It Works](#how-it-works)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running the Components](#running-the-components)
    - [Environment Variables](#environment-variables)
  - [Models \& Training](#models--training)
    - [Sentence Seq2Seq](#sentence-seq2seq)
    - [Single-Word Classifier](#single-word-classifier)
    - [Letter Classification (Fingerspelling)](#letter-classification-fingerspelling)
    - [Training Data](#training-data)
  - [Preprocessing](#preprocessing)
  - [Architecture](#architecture)
    - [Seq2Seq (multi\_attention)](#seq2seq-multi_attention)
    - [Classifier](#classifier)
  - [Desktop App (PySide6)](#desktop-app-pyside6)
  - [Build \& Deploy](#build--deploy)
  - [Known Issues](#known-issues)
  - [Roadmap](#roadmap)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

---

## How It Works

SignAI turns a short clip of someone signing into text. The desktop app is the
usual front door, but the same request-level flow applies wherever a video
reaches the inference API:

1. **Capture** — `app/camera.py` records webcam video in the desktop app.
   Pressing Record starts capture; pressing it again stops and hands the clip
   off for translation.
2. **Upload** — `app/api_call.py` sends the video to the local Flask API
   (`POST /api/upload` on `http://127.0.0.1:5000`).
3. **Preprocess** — `api/signai_api.py` saves the upload to
   `data/live/video/`, then `api/preprocessing_live_data.py` runs MediaPipe
   Holistic over every frame and writes the extracted keypoints to
   `data/live/live_dataset.csv`.
4. **Infer** — `api/inference.py` loads the trained `.keras` model (see
   [Environment Variables](#environment-variables) for how the model path is
   chosen) together with the gloss tokenizer, and predicts a translation
   with a confidence score.
5. **Display** — the API returns JSON to the app, which renders the result
   in the main window.

```
Webcam ──▶ camera.py ──▶ api_call.py ──▶ signai_api.py ──▶ preprocessing_live_data.py
                                                                    │
                                                                    ▼
                                              app UI  ◀── inference.py + gloss_tokenizer.json
```

The model itself — a BiLSTM encoder feeding an LSTM decoder with multi-head
attention — is trained separately ahead of time; see
[Models & Training](#models--training) and [Architecture](#architecture) for
how that training happens and what the network looks like.

---

## Getting Started

### Prerequisites

- **OS:** Windows (primary target). macOS/Linux support is in development
  and not yet verified end-to-end.
- **Python:** no version is pinned in `requirements.txt`, but the desktop
  build tooling (`app/builds/README.md`) documents and is tested against
  **Python 3.10–3.12**; that's the range to use unless you're prepared to
  debug version issues yourself.
- **Hardware:** a webcam for live recognition. A GPU is recommended for
  training and speeds up inference; CPU-only works but is slower.
- **Disk:** at least 5 GB free — trained models, caches, and MediaPipe
  assets add up quickly.

### Installation

```
git clone https://github.com/Stefanos0710/SignAI.git
cd SignAI
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions that are actually load-bearing here —
notably `tensorflow==2.16.2`, `keras==3.7.0`, `mediapipe==0.10.21`,
`protobuf==4.25.8`, and `numpy==1.26.4`. These aren't arbitrary: newer
protobuf or numpy releases break MediaPipe or TensorFlow compatibility, so
avoid upgrading them individually.

### Running the Components

Every command below assumes the virtual environment from the previous step
is active. The **cwd** column matters — several scripts resolve paths (like
`data/train_data`) relative to the current working directory, not to their
own location.

| Component | Command | Run from | Port |
|---|---|---|---|
| Desktop app | `python app.py` | `app/` | — (talks to the API internally) |
| Flask inference API | `python -m api.signai_api` | repo root | 5000 |
| Web API (Flask + SocketIO) | `python main.py` | repo root | 8000 (override with `PORT`) |
| Product website | `python main.py` | `product_webside/` | 5000, bound to `0.0.0.0` |
| Letter classification demo site | `python app.py` | `signai/letter_classification/website/` | 5000 |

Note that the Flask inference API and both demo websites default to the
same port (5000) — don't try to run more than one of them at a time without
changing the port in code, or you'll get a bind conflict.

The desktop app doesn't launch the API as a separate process you start
yourself; it imports and drives `api/signai_api.py` directly through
`app/api_call.py`. Running `python -m api.signai_api` by hand is mainly
useful for testing the API in isolation (e.g. with `curl` or Postman)
outside the desktop UI.

### Environment Variables

| Variable | Effect | Default |
|---|---|---|
| `SIGNAI_MODEL_PATH` / `SIGNAI_MODEL` | Overrides which `.keras` model the API loads | newest `models/trained_model_v*.keras` |
| `SIGNAI_DISABLE_SITE_CLEANUP` | Set to `1` to stop the app/API from stripping user site-packages off `sys.path` at startup | cleanup enabled |
| `PORT` | Port for the `main.py` Flask-SocketIO server | `8000` |

The site-packages cleanup exists because a system-wide protobuf install can
silently shadow the pinned `protobuf==4.25.8` from the venv and break
MediaPipe — disable it only if you're sure your environment doesn't have
that conflict. Desktop packaging has its own separate set of build-only
environment variables, documented in `app/builds/README.md`.

---

## Models & Training

### Sentence Seq2Seq

Primary translation model — BiLSTM encoder + LSTM decoder with 8-head MultiHeadAttention.

```
python signai/sentence_classification/train.py
```

Configuration is at the bottom of `signai/sentence_classification/train.py` (defaults: version 38.4, 200 epochs, batch 64, `multi_attention`).

**Key features:**
- Mixed precision training (`mixed_float16` global policy)
- Per-epoch WER, BLEU-1..4, ROUGE-1/2/L evaluation
- Epoch-wise augmentation (temporal: stretch/warp/freeze/dropout; spatial: shift/scale/rotate/noise), implemented in `signai/sentence_classification/augmentation.py`
- Transformer architecture also available in `signai/sentence_classification/experimental_transformer.py`

**Latest trained models:**

| Version | Type | Notes |
|---|---|---|
| v36 | BiLSTM-Seq2Seq | Latest internal version — June 2026|
| v30 | Seq2Seq | Latest public version — April 2026 |
| v29 | Seq2Seq | 200+ epochs, full history |
| v28 | Seq2Seq | 200+ epochs, full history |

- **Vocabulary:** 800+ gloss tokens
- **Output length:** Up to 15 tokens per sentence
- **Input features:** 426 per frame (7 pose + 21 left hand + 21 right hand + 93 face landmarks, each x/y/z)

### Single-Word Classifier

BiLSTM classifier using 150 features (pose + hands only, no face).

```
python signai/word_classification/train.py
```

Supports `--rebuild-cache` to force re-parsing of training CSVs.

| Metric | Value |
|---|---|
| Training accuracy | 99.8% |
| Validation accuracy | 98.7% |
| Architecture | BiLSTM(64) → BiLSTM(32) → Dense(64) → Softmax |

*Trained on a compressed subset of PHOENIX-Weather-2014T. Performance improves significantly with the full dataset.*

### Letter Classification (Fingerspelling)

An independent fingerspelling-alphabet classifier under `signai/letter_classification/` (previously the standalone `SignAlphaSet` sub-project). Has its own dataset, models, and a small Flask demo site — does not share code or training data with the sentence/word classifiers above.

```
python signai/letter_classification/train.py
```

Dataset download and preprocessing: `signai/letter_classification/download.py`, `preprocess_v2.py`/`preprocess_v3.py`. All scripts in this subtree must be run from the repo root — their data/model paths are hardcoded relative to it (e.g. `signai/letter_classification/data/...`).

### Training Data

Training CSVs (for sentence/word classification) are stored in `data/train_data/`. A parsed cache is kept at `.parsed_cache.pkl` — delete it or pass `--rebuild-cache` to re-parse. CSVs are git-ignored; only `example_for_train_data.csv` is tracked.

---

## Preprocessing

MediaPipe Holistic is used for keypoint extraction.

| Script | Purpose | Features | Landmarks |
|---|---|---|---|
| `signai/preprocessing/train_data.py` | Training data (sentence + word classification) | 426 (×3 xyz) | 7 pose + 42 hand + 93 face |
| `api/preprocessing_live_data.py` | Live inference | 151 (averaged) | 543 landmarks × 2 (xy) |

**Normalization pipeline:**

1. Video-wise shoulder midpoint centering
2. Shoulder-distance scaling
3. Savitzky–Golay temporal smoothing (window 9, polyorder 2)
4. Linear interpolation for missing keypoints

---

## Architecture

### Seq2Seq (multi_attention)

```
Encoder: Input(426) → Dense(1024) → LayerNorm → Dropout → DepthwiseConv1D → BiLSTM(512) → LayerNorm
Decoder: Embedding(256) → LayerNorm → LSTM(512) → LayerNorm → MultiHeadAttention(8 heads, residual) → Concat → Dense(512) → Dropout → LayerNorm → Dense(vocab, softmax)
```

### Classifier

```
Input(150) → Masking → BiLSTM(64) → Dropout(0.2) → BiLSTM(32) → Dropout(0.2) → Dense(64, ReLU) → Dropout(0.2) → Dense(classes, softmax)
```

---

## Desktop App (PySide6)

- **Workflow:** Press Record → perform signs → press again → upload to API → display translation
- **Result display:** `QPlainTextEdit`, hidden until ready, shows translation with optional debug info
- **Single-instance lock:** TCP port 52391
- **Logging:** stdout/stderr tee'd to `logs/desktop_app.log`
- **Settings:** `app/settings/settings.json`
- **Path handling:** `resource_path()` for bundled assets, `writable_path()` for per-user data (`%LOCALAPPDATA%\SignAI\`)
- **Qt fix:** `fix_qt_plugin_path()` must run before any PySide6 import
- **User-site cleanup:** User site-packages stripped from `sys.path` to avoid protobuf version conflicts
- **Build:** PyInstaller spec at `app/SignAI - Desktop.spec`, output at `build/SignAI - Desktop/SignAI - Desktop.exe`

---

## Build & Deploy

- **PyInstaller spec:** `app/SignAI - Desktop.spec` — bundles models, tokenizers, UI, icons (pathex set to repo root)
- **Updater:** `app/start_updater.py`, spec at `app/SignAI - Updater.spec`
- **Build scripts:** `app/builds/build-exe.py` (`--onefile`, `--include-models`, `--clean`, `--dry-run`), `build-updater-exe.py`, `build-final-app.py`, `build-zip.py` — see `app/builds/README.md` for the full release sequence and its own build-only environment variables
- **Runtime API overrides:** see [Environment Variables](#environment-variables)

---

## Known Issues

- **Camera feed:** If no image appears, press "Switch Camera" repeatedly. Close other camera-using apps.
- **Admin privileges:** Some operations may require elevation. Future releases will reduce this.
- **First-run delay:** Models load from disk on first launch — wait a few seconds for the UI to become responsive.
- **Recognition quality:** Degrades for casual or atypical signing. Addressed by planned augmentation and larger datasets.

---

## Roadmap

- Improve accuracy 3x via full datasets, larger compute, synthetic augmentation, and transformer architectures
- Expand vocabulary to thousands of gloss tokens
- Reduce admin access requirements
- Natural language rendering (gloss → grammatical sentences)
- Multilingual support (ASL planned)

---

## Contributing

1. Fork and create a branch: `git checkout -b feat/my-change`
2. Add tests and documentation for changes
3. Open a Pull Request with a clear description
4. Do **not** commit large model binaries — use release assets

---

## License

Non-commercial license. See [LICENSE](/LICENSE). Contact maintainers for alternative arrangements.

---

## Contact

- **General / press:** hello@signai.dev
- **Support:** open an issue at [GitHub Issues](https://github.com/Stefanos0710/SignAI/issues)
