# SignAI - Build Instructions

This folder contains all scripts and files needed to build SignAI as a standalone Windows executable (.exe).

## 📋 Prerequisites

Before building, make sure you have:
- Python 3.10-3.12 installed
- All dependencies installed: `pip install -r requirements.txt`
- PyInstaller installed: `pip install pyinstaller`
- At least 2GB free disk space

## 🚀 Quick Start

### Option 1: Using the Fixed Build Script (Recommended)

This is the easiest method with automatic Windows Defender handling:

1. **Right-click** on `build_exe_fixed.bat`
2. Select **"Run as Administrator"**
3. Follow the on-screen instructions
4. Wait for the build to complete (2-5 minutes)

The EXE will be created in: `app/dist/SignAI/SignAI.exe`

### Option 2: Using Python Script

```bash
cd app/builds
python build_exe.py
```

Follow the prompts to add Windows Defender exclusion if needed.

### Option 3: Manual Build

```bash
cd app
python -m PyInstaller --name=SignAI --windowed --onedir --noconfirm --clean --noupx --add-data="ui/main_window.ui;ui" --add-data="icons;icons" --add-data="style.qss;." --add-data="settings/settings.json;settings" --hidden-import=PySide6.QtCore --hidden-import=PySide6.QtGui --hidden-import=PySide6.QtWidgets --hidden-import=PySide6.QtUiTools --hidden-import=cv2 --hidden-import=mediapipe --hidden-import=numpy --hidden-import=requests --exclude-module=matplotlib --exclude-module=seaborn --exclude-module=pandas --exclude-module=scipy --exclude-module=tensorflow --exclude-module=keras app.py
```

## ⚠️ Windows Defender Issues

PyInstaller executables are often flagged by Windows Defender as false positives.

### Solution 1: Add Exclusion (Recommended)

1. Open **Windows Security**
2. Go to **Virus & Threat Protection**
3. Click **Manage Settings**
4. Scroll to **Exclusions**
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

