# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\ui\\main_window.ui', 'ui'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\icons', 'icons'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\style.qss', '.'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\settings\\settings.json', 'settings'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\tokenizers\\gloss_tokenizer.json', 'tokenizers'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\best_model_20251122_095715.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v28_epoch_01.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v28_epoch_02.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v28_epoch_03.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v28_epoch_04.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v28_epoch_05.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v28_epoch_06.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v28_epoch_07.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v28_epoch_08.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v29_epoch_01.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v29_epoch_02.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v29_epoch_03.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v29_epoch_04.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v29_epoch_05.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v29_epoch_06.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v29_epoch_07.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\checkpoint_v29_epoch_08.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\classifier_v23.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\classifier_v23_best.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\latest_best.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\latest_sign_classifier.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\sign_classifier_20251122_081326.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\sign_classifier_20251122_095042.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\sign_classifier_20251122_095715.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\sign_classifier_20251122_100735.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\sign_classifier_20251122_101016.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\sign_classifier_20251122_101301.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\sign_classifier_20251122_101813.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\sign_classifier_20251122_102028.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\sign_classifier_20251122_102204.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\sign_classifier_20251122_102938.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\sign_classifier_20251122_103141.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\trained_model_v18.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\trained_model_v19.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\trained_model_v20.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\trained_model_v21.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\trained_model_v22.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\trained_model_v23.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\trained_model_v24.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\trained_model_v25.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\trained_model_v27.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\trained_model_v28.fixed.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\trained_model_v28.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\trained_model_v29.keras', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_20251122_081326.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_20251122_095042.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_20251122_095715.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_20251122_100735.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_20251122_101016.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_20251122_101301.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_20251122_101813.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_20251122_102028.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_20251122_102204.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_20251122_102938.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_20251122_103141.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_history_v18.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_history_v19.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_history_v20.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_history_v21.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_history_v22.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_history_v23.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_history_v24.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_history_v25.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_history_v27.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_history_v28.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\models\\training_history_v29.png', 'models'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\videos', 'videos'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\videos\\history', 'videos\\history'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\data\\live', 'data\\live'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\data\\live\\video', 'data\\live\\video')]
binaries = []
hiddenimports = ['PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'PySide6.QtUiTools', 'PySide6.QtXml', 'PySide6.QtNetwork', 'PySide6.QtSvg', 'cv2', 'mediapipe', 'numpy', 'requests', 'psutil', 'flask', 'werkzeug', 'jinja2', 'threading', 'tensorflow', 'keras', 'ml_dtypes', 'pandas', 'scipy', 'json', 'csv', 'resource_path', 'camera', 'videos', 'settings', 'api_call', 'api.signai_api', 'api.preprocessing_live_data', 'api.inference', 'flask_cors', 'flask_socketio', 'matplotlib', 'matplotlib.pyplot', 'matplotlib.ft2font']
hiddenimports += collect_submodules('PySide6')
hiddenimports += collect_submodules('api')
hiddenimports += collect_submodules('PySide6')
tmp_ret = collect_all('PySide6')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mediapipe')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('flask')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('werkzeug')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('jinja2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('flask_cors')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('flask_socketio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('tensorflow')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('keras')
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
    excludes=[],
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
    uac_admin=True,
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
