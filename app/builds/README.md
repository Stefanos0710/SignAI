# SignAI - Build Instructions (Verbessert)

Dieser Ordner enthält alle Scripts und Dateien, die zum Erstellen von SignAI als eigenständige Windows-Anwendung benötigt werden.

## 📋 Voraussetzungen

Vor dem Build sicherstellen:
- **Python 3.10-3.12** installiert
- **Alle Dependencies installiert**: `pip install -r requirements.txt`
- **PyInstaller 6.16.0**: `pip install pyinstaller==6.16.0`
- Mindestens **4GB freier Festplattenspeicher**
- **Git** (für Updates)

## 🚀 Build-Optionen

### Option 1: Standard Build (Empfohlen)

Erstellt einen Ordner mit allen Dateien:

```bash
cd app\builds
python build-exe.py
```

**Ausgabe**: `app\dist\SignAI - Desktop\SignAI - Desktop.exe`

### Option 2: Single-File Build

Erstellt eine einzelne EXE-Datei (langsamer beim Start):

```bash
python build-exe.py --onefile
```

**Ausgabe**: `app\dist\SignAI - Desktop.exe`

### Option 3: Build mit allen Modellen

Inkludiert die AI-Modelle (größere Datei):

```bash
python build-exe.py --include-models
```

### Option 4: Minimal Build

Nur die notwendigen Dateien ohne API/Tokenizers:

```bash
python build-exe.py --no-include-api --no-include-tokenizers
```

## 🛠️ Build-Optionen

| Option | Beschreibung |
|--------|--------------|
| `--onedir` | Erstellt einen Ordner mit allen Dateien (Standard, schneller) |
| `--onefile` | Erstellt eine einzelne EXE-Datei |
| `--include-models` | Inkludiert AI-Modelle (~500MB) |
| `--include-tokenizers` | Inkludiert Tokenizers (Standard: aktiviert) |
| `--include-api` | Inkludiert API-Folder (Standard: aktiviert) |
| `--clean` | Löscht alte Build-Ordner vor dem Build |
| `--dry-run` | Zeigt Kommando ohne zu builden |

## 📦 Build-Prozess im Detail

### 1. Vorbereitung

```bash
# Alte Builds löschen
python build-exe.py --clean

# Dependencies prüfen
pip list | findstr "PySide6 tensorflow mediapipe opencv"
```

### 2. Build ausführen

```bash
# Standard Build
python build-exe.py

# Mit Cleanup
python build-exe.py --clean

# Test-Build (ohne Models)
python build-exe.py --clean
```

### 3. Nach dem Build

Der Build erstellt folgende Struktur:
```
app/
├── build/              # Temporäre Build-Dateien
└── dist/
    └── SignAI - Desktop/
        ├── SignAI - Desktop.exe  # Haupt-Anwendung
        ├── ui/                    # UI-Dateien
        ├── icons/                 # Icons
        ├── api/                   # API-Module
        ├── tokenizers/            # Tokenizers
        ├── videos/                # Video-Ordner
        └── ... (DLLs, Python-Libs, etc.)
```

## ⚠️ Häufige Probleme & Lösungen

### Problem 1: Windows Defender blockiert die EXE

**Lösung**: Exclusion hinzufügen

1. Windows Security öffnen
2. **Virus & Bedrohungsschutz** → **Einstellungen verwalten**
3. **Ausschlüsse** → **Ausschluss hinzufügen**
4. **Ordner** wählen und `app\dist` hinzufügen

**Oder**: Temporär deaktivieren während des Builds

### Problem 2: "ModuleNotFoundError" beim Ausführen

**Lösung**: Hidden Imports prüfen

```bash
# Build mit Debug-Info
python build-exe.py --onedir

# Log-Datei prüfen
type build\SignAI - Desktop\warn-SignAI - Desktop.txt
```

Fehlende Module zu `hidden_imports` in `build-exe.py` hinzufügen.

### Problem 3: TensorFlow/Keras funktioniert nicht

**Lösung**: 
- Stelle sicher, dass TensorFlow 2.16.2 installiert ist
- Prüfe ob `models/` Ordner vorhanden ist
- Verwende `--include-models` Option

### Problem 4: Mediapipe Fehler

**Lösung**: Mediapipe 0.10.14 verwenden

```bash
pip install mediapipe==0.10.14
```

### Problem 5: Kamera funktioniert nicht in der EXE

**Lösung**: OpenCV-Binaries prüfen

```bash
# Prüfen ob cv2 DLLs inkludiert sind
dir dist\SignAI - Desktop\.libs\cv2*
```

### Problem 6: UI wird nicht geladen

**Lösung**: UI-Dateien prüfen

```bash
# Prüfen ob UI-Dateien kopiert wurden
dir dist\SignAI - Desktop\ui\
```

Falls nicht vorhanden, `--add-data` in build-exe.py prüfen.

## 🧪 Testing nach dem Build

### 1. Basis-Test

```bash
# In den dist-Ordner wechseln
cd dist\SignAI - Desktop

# Anwendung starten
"SignAI - Desktop.exe"
```

### 2. Feature-Tests

- [ ] Kamera startet korrekt
- [ ] Video-Aufnahme funktioniert
- [ ] AI-Übersetzung funktioniert
- [ ] Settings werden gespeichert
- [ ] History wird gespeichert
- [ ] Updater startet

### 3. Performance-Test

- Startup-Zeit: < 10 Sekunden
- Kamera-Latenz: < 100ms
- AI-Response: < 5 Sekunden

## 📊 Build-Größen

| Build-Typ | Größe | Startup-Zeit |
|-----------|-------|--------------|
| `--onedir` (ohne Models) | ~800 MB | 3-5 Sek |
| `--onedir` (mit Models) | ~1.3 GB | 3-5 Sek |
| `--onefile` (ohne Models) | ~600 MB | 10-15 Sek |
| `--onefile` (mit Models) | ~1.1 GB | 15-20 Sek |

**Empfehlung**: `--onedir` für bessere Performance

## 🔧 Build-System Dateien

| Datei | Beschreibung |
|-------|--------------|
| `build-exe.py` | Haupt-Build-Script für Desktop-App |
| `build-updater-exe.py` | Build-Script für Updater |
| `build-final-app.py` | Kombiniert Desktop + Updater |
| `build-zip.py` | Erstellt Release-ZIP |
| `SignAI - Desktop.spec` | PyInstaller Spec-Datei |

## 🚢 Release erstellen

### Kompletter Release-Prozess

```bash
# 1. Desktop-App builden
python build-exe.py --clean

# 2. Updater builden
python build-updater-exe.py --clean

# 3. Final-Package erstellen
python build-final-app.py

# 4. ZIP für Distribution erstellen
python build-zip.py
```

### Oder alles auf einmal:

```bash
# Kompletter Build-Prozess
python build-exe.py --clean && ^
python build-updater-exe.py --clean && ^
python build-final-app.py && ^
python build-zip.py
```

## 📝 Notizen

### TensorFlow & GPU

- Standard-Build verwendet CPU-Version
- Für GPU-Support: `pip install tensorflow-gpu==2.16.2`
- GPU-Build benötigt CUDA 12.x und cuDNN

### Code Signing (Optional)

Für professionelle Distribution:

```bash
# Zertifikat erstellen (einmalig)
# Signiere die EXE nach dem Build
signtool sign /f certificate.pfx /p password "SignAI - Desktop.exe"
```

### Portable Version

Die `--onedir` Version ist bereits portabel:
- Kopiere den kompletten `dist\SignAI - Desktop` Ordner
- Keine Installation nötig
- Settings werden lokal gespeichert

## 🐛 Debug-Build

Für Entwicklung mit Console-Output:

```bash
# Entferne --noconsole Flag
python build-exe.py --console
```

Oder manuell in build-exe.py: Ändere `cmd = ["pyinstaller", "--noconsole"]` zu `cmd = ["pyinstaller", "--console"]`

## 📚 Weitere Ressourcen

- [PyInstaller Dokumentation](https://pyinstaller.org/en/stable/)
- [PySide6 Dokumentation](https://doc.qt.io/qtforpython/)
- [TensorFlow Freeze](https://www.tensorflow.org/guide/saved_model)

## 💡 Tipps für kleinere Builds

1. **Verwende `--exclude-module` für ungenutzte Pakten**:
   ```bash
   --exclude-module=tensorboard --exclude-module=matplotlib.tests
   ```

2. **Komprimiere mit UPX** (optional, kann Probleme verursachen):
   ```bash
   # In build-exe.py: Entferne --noupx
   ```

3. **Minimale Installation**:
   - Installiere nur benötigte Pakete
   - Verwende virtuelle Umgebung

## 🔄 Automatisierung

Erstelle `build-all.bat`:

```batch
@echo off
echo === SignAI Build System ===
echo.

echo [1/4] Cleaning old builds...
python build-exe.py --clean
if %errorlevel% neq 0 goto :error

echo.
echo [2/4] Building Desktop App...
python build-exe.py
if %errorlevel% neq 0 goto :error

echo.
echo [3/4] Building Updater...
python build-updater-exe.py
if %errorlevel% neq 0 goto :error

echo.
echo [4/4] Creating Final Package...
python build-final-app.py
if %errorlevel% neq 0 goto :error

echo.
echo === Build Complete! ===
echo Output: app\final\SignAI - Desktop\
pause
exit /b 0

:error
echo.
echo === Build Failed! ===
pause
exit /b 1
```

## ✅ Checklist vor Release

- [ ] Alle Tests bestanden
- [ ] Version-Nummer aktualisiert
- [ ] README aktualisiert
- [ ] CHANGELOG aktualisiert
- [ ] License-Dateien vorhanden
- [ ] Icons korrekt
- [ ] Keine Debug-Logs in Produktion
- [ ] Performance getestet
- [ ] Auf verschiedenen Windows-Versionen getestet
- [ ] Installer getestet (falls verwendet)

## 📧 Support

Bei Problemen:
1. Prüfe die Logs in `build/SignAI - Desktop/`
2. Verwende `--dry-run` um das Kommando zu sehen
3. Öffne ein Issue auf GitHub mit Details

---

**Happy Building! 🚀**
5. Click **Add or remove exclusions**
6. Click **Add an exclusion** → **Folder**
7. Select: `C:\Users\<YourUser>\Documents\GitHub\SignAI\app`

### Solution 2: Temporarily Disable (Not Recommended)

Only use this if you trust the code:
1. Open Windows Security
2. Go to Virus & Threat Protection
3. Turn off Real-time protection temporarily
4. Run the build script
5. Re-enable protection after building

## 📁 Output Structure

After successful build:

```
app/
├── builds/           # Build scripts (you are here)
├── dist/
│   └── SignAI/       # ← Your executable folder
│       ├── SignAI.exe    # Main executable
│       ├── icons/        # App icons
│       ├── ui/           # UI files
│       ├── settings/     # Settings
│       └── [DLLs]        # Required libraries
└── build/            # Temporary build files (can be deleted)
```

⚠️ **Important**: The entire `dist/SignAI/` folder is required to run the app!

## 🎯 Running the Application

After building:

```bash
# Navigate to the output folder
cd app/dist/SignAI

# Run the executable
SignAI.exe
```

Or simply double-click `SignAI.exe` in the file explorer.

## 📦 Distribution

To share the application:

1. Zip the entire `dist/SignAI/` folder
2. Share the zip file
3. Users extract and run `SignAI.exe`

## 🔧 Build Scripts Explained

### `build_exe_fixed.bat`
- Full-featured build script
- Automatically attempts to add Windows Defender exclusion
- Activates virtual environment
- Best for regular use

### `build_exe.py`
- Python version of the build script
- Interactive prompts
- Cross-platform compatible
- Good for development

### `SignAI.spec`
- PyInstaller specification file
- Defines what to include/exclude
- Can be customized for advanced builds

## 🐛 Troubleshooting

### Build Fails with "pyinstaller not found"

```bash
pip install pyinstaller
```

### Build Fails with Windows Defender Error 225

Follow the Windows Defender exclusion steps above.

### EXE Crashes Immediately

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Missing UI or Icons

The build script automatically includes these. If missing, check that:
- `ui/main_window.ui` exists
- `icons/` folder exists
- You're running the build from the correct directory

### EXE is Too Large

The EXE includes all dependencies. To reduce size:
- Remove unused modules from `--exclude-module` list
- Use `--onefile` instead of `--onedir` (slower startup)

## 📝 Customization

### Change Icon

Edit `SignAI.spec` and add:
```python
icon='path/to/your/icon.ico'
```

Then run:
```bash
pyinstaller SignAI.spec
```

### Include Additional Files

Edit the `--add-data` parameters in the build scripts:
```bash
--add-data="your_file.txt;."
```

### Exclude More Modules

Add to the `--exclude-module` list to reduce size:
```bash
--exclude-module=module_name
```

## 📊 Build Statistics

Typical build results:
- **Build Time**: 2-5 minutes
- **Output Size**: ~500-800 MB (includes all dependencies)
- **Startup Time**: 2-4 seconds
- **Platforms**: Windows 10/11 (x64)

## 🆘 Support

If you encounter issues:
1. Check the error messages carefully
2. Ensure all prerequisites are met
3. Try running as Administrator
4. Check [GitHub Issues](https://github.com/Stefanos0710/SignAI/issues)

## 📄 License

This build process is part of the SignAI project.
See the main LICENSE file for details.

---

**Last Updated**: 2025-10-19  
**Author**: Stefanos Koufogazos Loukianov

