# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\app.py'],
    pathex=['C:\\Users\\stefa\\Documents\\GitHub\\SignAI'],
    binaries=[],
    datas=[('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\ui', 'ui'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\icons', 'icons'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\videos', 'videos'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\style.qss', '.'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\settings', 'settings'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\app\\version.txt', '.'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\tokenizers', 'tokenizers'), ('C:\\Users\\stefa\\Documents\\GitHub\\SignAI\\api', 'api')],
    hiddenimports=['numpy.core._methods', 'numpy.lib.format', 'numpy._globals', 'numpy._distributor_init', 'cv2', 'numpy'],
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
