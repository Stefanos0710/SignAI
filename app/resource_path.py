"""

this should be in every file

"""

import os, sys

def resource_path(relative_path):
    # Prefer the directory of the executable (onedir build)
    if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.abspath('.')

    full = os.path.join(base, relative_path)
    if os.path.exists(full):
        return full

    # Fallback for legacy onefile builds: use PyInstaller temp extraction (_MEIPASS)
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        alt = os.path.join(meipass, relative_path)
        if os.path.exists(alt):
            return alt

    # Return best-effort path (may be missing) to surface a clear error
    return full


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
