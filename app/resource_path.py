"""

this should be in every file

"""

import os, sys


def _base_dir() -> str:
    """Folder where bundled resources live."""
    if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _user_data_dir() -> str:
    """Per-user writable directory to avoid permission issues (Program Files, etc.)."""
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(local, "SignAI")


def resource_path(relative_path: str) -> str:
    base = _base_dir()
    full = os.path.join(base, relative_path)
    if os.path.exists(full):
        return full

    # Fallback for onefile builds (_MEIPASS extraction)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        alt = os.path.join(meipass, relative_path)
        if os.path.exists(alt):
            return alt

    return full


def writable_path(relative_path: str) -> str:
    # Always use per-user data dir when frozen to avoid admin rights problems on other PCs.
    if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
        base = _user_data_dir()
    else:
        base = _base_dir()

    full = os.path.join(base, relative_path)
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return full
