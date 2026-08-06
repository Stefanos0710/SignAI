# SignAI — Build Instructions

Scripts and files needed to package SignAI as a standalone Windows application.

## Prerequisites

- Python 3.10–3.12
- All dependencies installed: `pip install -r requirements.txt`
- PyInstaller 6.16.0: `pip install pyinstaller==6.16.0`
- At least 4 GB free disk space
- Git (for updates)

## Build options

### Standard build (recommended)

Produces a folder with all files:

```bash
cd app\builds
python build-exe.py
```

Output: `app\dist\SignAI - Desktop\SignAI - Desktop.exe`

### Single-file build

Produces a single EXE (slower startup):

```bash
python build-exe.py --onefile
```

### Build with all models

Includes the AI models (larger file):

```bash
python build-exe.py --include-models
```

### Minimal build

Only the essential files, no API/tokenizers:

```bash
python build-exe.py --no-include-api --no-include-tokenizers
```

| Flag | Description |
|---|---|
| `--onedir` | Folder output (default, faster startup) |
| `--onefile` | Single EXE output |
| `--include-models` | Include AI models (~500 MB) |
| `--include-tokenizers` | Include tokenizers (default: on) |
| `--include-api` | Include the `api/` folder (default: on) |
| `--clean` | Delete old build folders first |
| `--dry-run` | Print the PyInstaller command without building |

## Build system files

| File | Purpose |
|---|---|
| `build-exe.py` | Main build script for the desktop app |
| `build-updater-exe.py` | Build script for the updater |
| `build-final-app.py` | Combines desktop app + updater into a release package |
| `build-zip.py` | Creates the release ZIP |
| `runtime_qt_plugin_path.py` | Qt plugin path fix injected into frozen builds |
| `manifest-admin.py` | Generates the admin-elevation manifest |
| `../SignAI - Desktop.spec` | PyInstaller spec for the desktop app |
| `../SignAI - Updater.spec` | PyInstaller spec for the updater |
| `*.ifp` | InstallForge installer project files |

## Full release process

```bash
python build-exe.py --clean
python build-updater-exe.py --clean
python build-final-app.py
python build-zip.py
```

Output structure after a successful build:

```
app/
├── build/              # temporary PyInstaller build files (safe to delete)
└── dist/
    └── SignAI - Desktop/
        ├── SignAI - Desktop.exe
        ├── ui/
        ├── icons/
        ├── api/
        ├── tokenizers/
        ├── videos/
        └── ... (DLLs, Python libs, etc.)
```

## Testing after a build

- [ ] Camera starts correctly
- [ ] Video recording works
- [ ] AI translation works
- [ ] Settings persist
- [ ] History is saved
- [ ] Updater starts

Startup should be under ~10 seconds and translation under ~5 seconds.

## Troubleshooting

**Windows Defender blocks the EXE** — add `app\dist` as an exclusion (Windows Security → Virus & threat protection → Manage settings → Add or remove exclusions → Folder), or temporarily disable real-time protection during the build only.

**`ModuleNotFoundError` at runtime** — check `build\SignAI - Desktop\warn-SignAI - Desktop.txt` for missing modules and add them to `hidden_imports` in `build-exe.py`.

**TensorFlow/Keras doesn't work** — confirm TensorFlow 2.16.2 is installed and `models/` exists; use `--include-models` if the build needs bundled models.

**MediaPipe errors** — this project pins MediaPipe 0.10.21; don't substitute a different version.

**Camera doesn't work in the built EXE** — check that OpenCV's DLLs were bundled (`dir dist\SignAI - Desktop\.libs\cv2*`).

**UI doesn't load** — verify `ui/main_window.ui` and `icons/` exist and that the build was run from `app\builds`.

**Build is too large** — add more `--exclude-module` entries (e.g. `tensorboard`, `matplotlib.tests`), or avoid `--include-models` and ship models separately.

## Notes

- Default build targets CPU-only TensorFlow. GPU builds need `tensorflow-gpu==2.16.2` plus CUDA 12.x / cuDNN, and are not part of the standard release process.
- The `--onedir` output is already portable — copy the whole `dist\SignAI - Desktop` folder, no installer required.
- For code signing: `signtool sign /f certificate.pfx /p password "SignAI - Desktop.exe"`.
