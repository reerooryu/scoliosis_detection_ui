# Where the app and its files are, both when run from source and when run as
# the packaged app (PyInstaller sets sys.frozen).

import os
import sys
from pathlib import Path

from config import LOCAL_MODEL_FILENAME

SOURCE_DIR = Path(__file__).resolve().parent.parent
# The model file inside a source checkout (not in git; see README).
SOURCE_MODEL_PATH = SOURCE_DIR / "model" / "model_t001_6_effb5_mask_kp_2cls" / "model_final_run.pth"


def _log_dir():
    """%LOCALAPPDATA%\\ScoliosisAI on Windows, ~/Library/Logs/ScoliosisAI on
    a Mac, ~/.cache/ScoliosisAI elsewhere."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "ScoliosisAI"


# The app's log, replaced at each start (app.py). Testers can send it when
# something goes wrong.
LOG_PATH = _log_dir() / "log.txt"


def is_packaged():
    return bool(getattr(sys, "frozen", False))


def find_model_file():
    """A model file found without asking the user, or None.

    Packaged app: ScoliosisAI_model.pth in the app's folder or a few folders
    above it (on a Mac the program is inside ScoliosisAI.app/Contents/MacOS,
    so a file next to ScoliosisAI.app is three folders up).
    From source: the model folder of the checkout.
    """
    if is_packaged():
        folder = Path(sys.executable).resolve().parent
        for candidate_dir in [folder, *folder.parents[:4]]:
            candidate = candidate_dir / LOCAL_MODEL_FILENAME
            if candidate.is_file():
                return str(candidate)
        return None
    for candidate in (SOURCE_MODEL_PATH, SOURCE_DIR / LOCAL_MODEL_FILENAME):
        if candidate.is_file():
            return str(candidate)
    return None
