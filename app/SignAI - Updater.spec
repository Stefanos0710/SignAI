# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\updater', 'updater'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\updater\\updater_app.ui', '.'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\ui', 'ui'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\icons', 'icons'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\updater\\style.qss', '.'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\settings', 'settings'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\updater\\version.txt', '.')]
binaries = []
hiddenimports = ['numpy', 'cv2', 'PySide6.QtWidgets', 'PySide6.QtUiTools', 'PySide6.QtCore', 'requests', 'dotenv', 'matplotlib.ft2font', 'matplotlib.backends', 'matplotlib.pyplot', 'matplotlib.cbook', 'matplotlib._c_internal_utils', 'matplotlib._api', 'matplotlib._docstring', 'matplotlib._pylab_helpers']
hiddenimports += collect_submodules('matplotlib')
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mediapipe')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('tensorflow')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\updater\\updater-app.py'],
    pathex=['C:\\Users\\stefa\\Documents\\GitHub\\SignAI'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
