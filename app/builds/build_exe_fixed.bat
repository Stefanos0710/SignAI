@echo off
REM ============================================
REM SignAI EXE Builder - Windows Defender Fix
REM ============================================

echo ============================================
echo SignAI - EXE Builder
echo ============================================
echo.
echo IMPORTANT: Windows Defender blocks PyInstaller!
echo.
echo Please perform ONE of the following steps:
echo.
echo Option 1: Run as Administrator
echo   - Right-click on this file
echo   - Select "Run as Administrator"
echo.
echo Option 2: Add Windows Defender exclusion manually
echo   1. Open Windows Security
echo   2. Go to Virus ^& Threat Protection
echo   3. Manage Settings
echo   4. Under "Exclusions" click "Add exclusion"
echo   5. Select "Folder" and add:
echo      %CD%
echo   6. Run this file again
echo.
echo Option 3: Temporarily disable Windows Defender
echo   (Not recommended, but works)
echo.
echo ============================================
pause

REM Try to add Windows Defender exclusion (requires Admin)
echo.
echo Attempting to add Windows Defender exclusion...
powershell -Command "Start-Process powershell -ArgumentList 'Add-MpPreference -ExclusionPath ''%CD%''' -Verb RunAs -Wait" 2>nul
if %errorlevel% equ 0 (
    echo Exclusion successfully added!
    timeout /t 2 >nul
) else (
    echo Could not automatically add exclusion.
    echo Please add manually or run as Administrator.
    echo.
    pause
)

REM Activate Virtual Environment
echo.
echo Activating Virtual Environment...
if exist "C:\Users\stefa\PycharmProjects\SignAI\DropSend\.venv\Scripts\activate.bat" (
    call "C:\Users\stefa\PycharmProjects\SignAI\DropSend\.venv\Scripts\activate.bat"
    echo Virtual Environment activated!
) else if exist "..\..\.venv\Scripts\activate.bat" (
    call "..\..\.venv\Scripts\activate.bat"
    echo Virtual Environment activated!
) else (
    echo WARNING: Virtual Environment not found!
    echo PyInstaller must be installed system-wide.
    pause
)

REM Delete old build files
echo.
echo Deleting old build files...
if exist build rd /s /q build 2>nul
if exist dist rd /s /q dist 2>nul
if exist __pycache__ rd /s /q __pycache__ 2>nul
if exist SignAI.spec del /q SignAI.spec 2>nul

REM Create EXE without icon (bypasses Windows Defender issue)
echo.
echo Creating EXE file (without icon to bypass Windows Defender)...
echo.

pyinstaller --name=SignAI ^
    --windowed ^
    --onedir ^
    --noconfirm ^
    --clean ^
    --noupx ^
    --add-data="ui/main_window.ui;ui" ^
    --add-data="icons;icons" ^
    --add-data="style.qss;." ^
    --add-data="settings/settings.json;settings" ^
    --hidden-import=PySide6.QtCore ^
    --hidden-import=PySide6.QtGui ^
    --hidden-import=PySide6.QtWidgets ^
    --hidden-import=PySide6.QtUiTools ^
    --hidden-import=cv2 ^
    --hidden-import=mediapipe ^
    --hidden-import=numpy ^
    --hidden-import=requests ^
    --exclude-module=matplotlib ^
    --exclude-module=seaborn ^
    --exclude-module=pandas ^
    --exclude-module=scipy ^
    --exclude-module=tensorflow ^
    --exclude-module=keras ^
    --exclude-module=torch ^
    --exclude-module=torchvision ^
    --exclude-module=torchaudio ^
    app.py

echo.
echo ============================================
if exist dist\SignAI\SignAI.exe (
    echo.
    echo ╔════════════════════════════════════════╗
    echo ║   SUCCESSFULLY CREATED!                ║
    echo ╚════════════════════════════════════════╝
    echo.
    echo The EXE file is located in: dist\SignAI\
    echo.
    echo To start the app:
    echo   dist\SignAI\SignAI.exe
    echo.
    echo NOTE: The entire dist\SignAI\ folder is required!
    echo.
) else (
    echo.
    echo ╔════════════════════════════════════════╗
    echo ║   ERROR creating the EXE!              ║
    echo ╚════════════════════════════════════════╝
    echo.
    echo Possible solutions:
    echo 1. Run as Administrator
    echo 2. Add Windows Defender exclusion
    echo 3. Temporarily disable Windows Defender
    echo 4. Wait 1-2 minutes and try again
    echo.
)
echo ============================================
echo.
pause
