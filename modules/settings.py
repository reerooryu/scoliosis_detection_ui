# Saved settings: where the AI model runs, curve colors and the default
# export folder. Stored with QSettings, so they survive restarts. This module
# only reads and writes them; the Settings window (modules/settings_dialog.py)
# shows them.

import os
from dataclasses import dataclass

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor

from config import (
    CURVE_COLORS, DEFAULT_INFERENCE_MODE, FORCED_API_URL, FORCED_INFERENCE_MODE, INFERENCE_API_URL,
)
from modules.app_paths import find_model_file

ORG_NAME = "ScoliosisSuite"
APP_KEY = "DetectionUI"

MODE_LOCAL = "local"
MODE_SERVER = "server"

# Language switch (disabled until Thai translations exist): stored as
# "language", "English" or "Thai". See the note in settings_dialog.py.


@dataclass(frozen=True)
class InferenceSettings:
    """Where the AI model runs, as currently saved (or forced by config)."""
    mode: str          # MODE_LOCAL or MODE_SERVER
    api_url: str       # server mode
    weights_path: str  # local mode; "" if no model file is chosen or found


def _store():
    return QSettings(ORG_NAME, APP_KEY)


def inference_settings():
    """Where the AI model runs. FORCED_* values in config.py win over what
    the user saved."""
    store = _store()
    mode = FORCED_INFERENCE_MODE or store.value("inference_mode", DEFAULT_INFERENCE_MODE)
    if mode not in (MODE_LOCAL, MODE_SERVER):
        mode = DEFAULT_INFERENCE_MODE
    api_url = FORCED_API_URL or store.value("inference_api_url", INFERENCE_API_URL)
    # A saved file that has since been moved or deleted is ignored, so a
    # model file placed next to the app can still be found.
    weights_path = store.value("weights_path", "")
    if not weights_path or not os.path.isfile(weights_path):
        weights_path = find_model_file() or ""
    return InferenceSettings(mode=mode, api_url=api_url, weights_path=weights_path)


def save_inference_settings(mode, api_url, weights_path):
    """Save where the AI model runs. Values forced in config.py are not
    saved; a blank address falls back to the default; an empty model path
    leaves the saved one unchanged."""
    store = _store()
    if not FORCED_INFERENCE_MODE:
        store.setValue("inference_mode", mode)
    if not FORCED_API_URL:
        store.setValue("inference_api_url", api_url.strip() or INFERENCE_API_URL)
    if weights_path:
        store.setValue("weights_path", weights_path)


def save_weights_path(path):
    """Remember the model file (for example, chosen at the first start)."""
    _store().setValue("weights_path", path)


def curve_colors():
    """The color of each curve as hex strings, top curve first. Anything
    missing or invalid falls back to the default."""
    store = _store()
    colors = []
    for index, default in enumerate(CURVE_COLORS):
        saved = store.value("curve_color_{}".format(index + 1))
        if saved is None and index == 0:
            # Older versions had one line color; keep it for curve 1.
            saved = store.value("line_color")
        color = QColor(saved) if saved else QColor()
        colors.append(color.name() if color.isValid() else default)
    return colors


def save_curve_colors(colors):
    store = _store()
    for index, color in enumerate(colors):
        store.setValue("curve_color_{}".format(index + 1), QColor(color).name())


def export_folder():
    return _store().value("export_folder", os.path.expanduser("~"))


def save_export_folder(folder):
    _store().setValue("export_folder", folder)
