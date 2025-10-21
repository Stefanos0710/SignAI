"""

this should be in every file

"""

import os, sys

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)


def writable_path(relative_path):
    # If frozen by PyInstaller, prefer directory next to the executable
    if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
        base = os.path.dirname(sys.executable)
    else:
        # use project working dir (where the developer runs the script)
        base = os.path.abspath('.')

    full = os.path.join(base, relative_path)
    # ensure parent directories exist for writable paths
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return full
