# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\ui', 'ui'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\icons', 'icons'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\videos', 'videos'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\style.qss', '.'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\settings', 'settings'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\updater', 'updater'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\tokenizers', 'tokenizers'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\api', 'api')]
binaries = [('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\.venv\\Lib\\site-packages\\numpy\\core', 'numpy/core'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\.venv\\Lib\\site-packages\\tensorflow\\python', 'tensorflow/python'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\.venv\\Lib\\site-packages\\mediapipe\\modules', 'mediapipe/modules')]
hiddenimports = ['PySide6.QtWidgets', 'PySide6.QtUiTools', 'PySide6.QtCore', 'PySide6.QtGui', 'camera', 'settings', 'videos', 'api_call', 'resource_path', 'api.signai_api', 'api.inference', 'api.preprocessing_live_data', 'tensorflow', 'tensorflow.python', 'tensorflow.python.platform._pywrap_tf2', 'tensorflow.python.framework.ops', 'tensorflow.python.trackable', 'tensorflow.python.trackable.data_structures', 'tensorflow.python.trackable.base', 'tensorflow.python.training.tracking', 'tensorflow.python.eager', 'tensorflow.python.saved_model', 'tensorflow.python.keras', 'tensorflow.python.keras.saving', 'tensorflow.keras', 'tensorflow.keras.models', 'tensorflow.keras.layers', 'keras', 'keras.models', 'keras.layers', 'keras.saving', 'mediapipe', 'mediapipe.python', 'mediapipe.python.solutions', 'mediapipe.python.solutions.holistic', 'matplotlib', 'matplotlib.backends', 'matplotlib.backends.backend_agg', 'matplotlib._c_internal_utils', 'matplotlib.ft2font', 'matplotlib.pyplot', 'matplotlib.cbook', 'matplotlib._api', 'matplotlib._docstring', 'matplotlib._pylab_helpers', 'numpy', 'numpy.core._methods', 'numpy.lib.format', 'numpy._globals', 'numpy._distributor_init', 'cv2', 'flask', 'flask_cors', 'werkzeug', 'werkzeug.utils', 'jinja2', 'click', 'pandas', 'scipy', 'requests', 'psutil', 'threading', 'json', 'pickle']
datas += collect_data_files('mediapipe')
hiddenimports += collect_submodules('mediapipe.python')
tmp_ret = collect_all('tensorflow')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('keras')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('flask')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\app.py'],
    pathex=['C:\\Users\\stefa\\Documents\\GitHub\\SignAI', 'C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\builds'],
    hooksconfig={},
    runtime_hooks=['C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\builds\\pyinstaller_hook_tensorflow.py', 'C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\builds\\pyinstaller_runtime_hook.py'],
    excludes=['mediapipe.tasks', 'mediapipe.tasks.python', 'mediapipe.tasks.python.genai', 'mediapipe.tasks.python.genai.converter', 'mediapipe.tasks.python.audio', 'mediapipe.tasks.python.audio.audio_classifier', 'mediapipe.tasks.python.audio.core', 'mediapipe.tasks.python.audio.core.base_audio_task_api', 'mediapipe.tasks.python.core', 'mediapipe.tasks.python.core.optional_dependencies', 'mediapipe.tasks.python.vision', 'mediapipe.tasks.python.text', 'torch', 'tensorboard', 'tensorflow.tools', 'tensorflow.tools.docs', 'tensorflow.tools.docs.doc_controls', 'tensorflow.python.debug', 'tensorflow.lite.python.lite'],
    noarchive=False,
    optimize=2,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [('O', None, 'OPTION'), ('O', None, 'OPTION')],
    exclude_binaries=True,
    name='SignAI - Desktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
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
    upx=False,
    upx_exclude=[],
    name='SignAI - Desktop',
)
