# app/builds/runtime_qt_plugin_path.py
import os
import sys

def _set_qt_env():
    # Basisverzeichnis für gefrorene App
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)

    candidates = [
        os.path.join(base, "PySide6", "plugins"),
        os.path.join(base, "_internal", "PySide6", "plugins"),
    ]

    plugins = next((p for p in candidates if os.path.isdir(p)), None)
    if not plugins:
        print("[Hook] No PySide6 plugin dir found")
        return

    platforms = os.path.join(plugins, "platforms")

    os.environ["QT_PLUGIN_PATH"] = plugins
    if os.path.isdir(platforms):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platforms

    print("[Hook] QT_PLUGIN_PATH:", plugins)
    if os.path.isdir(platforms):
        print("[Hook] QT_QPA_PLATFORM_PLUGIN_PATH:", platforms)

_set_qt_env()
