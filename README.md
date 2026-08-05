<div align="center">

<img src="app/icons/icon.png" alt="SignAI" width="130"/>

# SignAI — Sign Language Translator

**Real-time Sign Language recognition and gloss translation using deep learning**

<br>


[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
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

- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Models & Training](#models--training)
- [Preprocessing](#preprocessing)
- [Architecture](#architecture)
- [Desktop App](#desktop-app-pyside6)
- [Build & Deploy](#build--deploy)
- [Known Issues](#known-issues)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Quick Start

| Component | Command |
|---|---|
| Desktop app | `cd app && python app.py` |
| Web API (Flask + SocketIO, port 8000) | `python main.py` |
| Flask API (port 5000) | `python -m api.signai_api` |
| API client (background server + upload) | `python -c "import request; request.start('video.mp4')"` |
| Product website | `cd product_webside && python main.py` |

```
pip install -r requirements.txt
```

---

## Requirements

- **OS:** Windows (primary). macOS/Linux support in development.
- **Hardware:** Webcam for live recognition. GPU recommended; CPU-only supported but slower.
- **Disk:** 5 GB minimum (models and caches require more).
- **Python:** 3.8+
- **Core stack:** TensorFlow 2.16.2 / Keras 3.7.0, MediaPipe 0.10.21, numpy 1.26.4, protobuf 4.25.8

---

## Models & Training

### Sentence Seq2Seq

Primary translation model — BiLSTM encoder + LSTM decoder with 8-head MultiHeadAttention.

```
python train-seq2seq.py
```

Configuration is at the bottom of `train-seq2seq.py` (defaults: version 38.4, 200 epochs, batch 64, `multi_attention`).

**Key features:**
- Mixed precision training (`mixed_float16` global policy)
- Per-epoch WER, BLEU-1..4, ROUGE-1/2/L evaluation
- Epoch-wise augmentation (temporal: stretch/warp/freeze/dropout; spatial: shift/scale/rotate/noise)
- Transformer architecture also available in `utils_experimental_train.py`

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
python train.py
```

Supports `--rebuild-cache` to force re-parsing of training CSVs.

| Metric | Value |
|---|---|
| Training accuracy | 99.8% |
| Validation accuracy | 98.7% |
| Architecture | BiLSTM(64) → BiLSTM(32) → Dense(64) → Softmax |

*Trained on a compressed subset of PHOENIX-Weather-2014T. Performance improves significantly with the full dataset.*

### Training Data

Training CSVs are stored in `data/train_data/`. A parsed cache is kept at `.parsed_cache.pkl` — delete it or pass `--rebuild-cache` to re-parse. CSVs are git-ignored; only `example_for_train_data.csv` is tracked.

---

## Preprocessing

MediaPipe Holistic is used for keypoint extraction.

| Script | Purpose | Features | Landmarks |
|---|---|---|---|
| `preprocessing_train_data.py` | Training data | 426 (×3 xyz) | 7 pose + 42 hand + 93 face |
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
- **API overrides:** `SIGNAI_MODEL_PATH` / `SIGNAI_MODEL` for custom model path, `SIGNAI_DISABLE_SITE_CLEANUP=1` to disable user-site cleanup

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
