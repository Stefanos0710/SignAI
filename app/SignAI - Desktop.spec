# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\ui', 'ui'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\icons', 'icons'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\videos', 'videos'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\style.qss', '.'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\settings', 'settings'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\updater', 'updater'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\tokenizers', 'tokenizers'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\api', 'api'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models', 'models')]
binaries = []
hiddenimports = ['PySide6.QtWidgets', 'PySide6.QtUiTools', 'PySide6.QtCore', 'tensorflow.python.platform._pywrap_tf2', 'tensorflow.python', 'tensorflow.python.framework.ops', 'tensorflow.python.trackable', 'tensorflow.python.trackable.data_structures', 'tensorflow.python.trackable.base', 'tensorflow.python.training.tracking', 'mediapipe', 'matplotlib._c_internal_utils', 'matplotlib.ft2font', 'matplotlib.backends', 'matplotlib.pyplot', 'matplotlib.cbook', 'matplotlib._api', 'matplotlib._docstring', 'matplotlib._pylab_helpers', 'numpy.core._methods', 'numpy.lib.format', 'numpy._globals', 'numpy._distributor_init', 'cv2', 'numpy']
hiddenimports += collect_submodules('tensorflow')
hiddenimports += collect_submodules('mediapipe')
hiddenimports += collect_submodules('matplotlib')
tmp_ret = collect_all('tensorflow')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mediapipe')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\app.py'],
    pathex=['C:\\Users\\stefa\\Documents\\GitHub\\SignAI'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['mediapipe.tasks.python.genai.converter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SignAI - Desktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\icons\\icon.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SignAI - Desktop',
)
