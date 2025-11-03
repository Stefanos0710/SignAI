# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\updater\\updater-app.py'],
    pathex=['C:\\Users\\stefa\\Documents\\GitHub\\SignAI'],
    binaries=[],
    datas=[('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\updater', 'updater'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\updater\\updater_app.ui', '.'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\ui', 'ui'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\icons', 'icons'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\updater\\style.qss', '.'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\settings', 'settings'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\updater\\version.txt', '.')],
    hiddenimports=['PySide6.QtWidgets', 'PySide6.QtUiTools', 'PySide6.QtCore', 'requests', 'dotenv', 'matplotlib.ft2font'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SignAI - Updater',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\icons\\icon.png'],
)
