# Settings dialog: where the AI model runs (this computer or a server),
# curve colors and default export folder. Saved with QSettings, so they
# survive restarts.

import os
from dataclasses import dataclass

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QColorDialog, QDialogButtonBox, QRadioButton,
    QButtonGroup, QGroupBox,
    # QComboBox,  # only needed for the language switch below -- re-add on import if re-enabling
)

from config import (
    CURVE_COLORS, DEFAULT_INFERENCE_MODE, FORCED_API_URL, FORCED_INFERENCE_MODE, INFERENCE_API_URL,
)
from modules.app_paths import find_model_file

ORG_NAME = "ScoliosisSuite"
APP_KEY = "DetectionUI"

# --- Language switch (disabled until Thai translations exist) --------------
# To re-enable: uncomment this block, the QComboBox import above, the
# "Language:" row in __init__, the language line in _on_save, and
# get_saved_language() below.
# LANGUAGES = ["English", "Thai"]
# DEFAULT_LANGUAGE = "English"
# ---------------------------------------------------------------------------

MODE_LOCAL = "local"
MODE_SERVER = "server"
MODEL_FILE_FILTER = "AI model file (*.pth);;All files (*)"


@dataclass(frozen=True)
class InferenceSettings:
    """Where the AI model runs, as currently saved (or forced by config)."""
    mode: str          # MODE_LOCAL or MODE_SERVER
    api_url: str       # server mode
    weights_path: str  # local mode; "" if no model file is chosen or found


def choose_model_file(parent, start_path=""):
    """Ask for the model file. Returns its path, or "" if cancelled."""
    path, _ = QFileDialog.getOpenFileName(
        parent, "Choose the AI Model File", start_path or os.path.expanduser("~"), MODEL_FILE_FILTER
    )
    return path


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(460)

        self._settings = QSettings(ORG_NAME, APP_KEY)
        self._curve_colors = [QColor(color) for color in self.get_saved_curve_colors()]
        self._export_folder = self._settings.value("export_folder", os.path.expanduser("~"))
        inference = self.get_inference_settings()
        self._weights_path = inference.weights_path

        model_box = self._build_model_box(inference)

        form = QFormLayout()

        # One button per curve, filled with that curve's color. Click to change.
        color_row = QHBoxLayout()
        self._color_buttons = []
        for index in range(len(CURVE_COLORS)):
            button = QPushButton(str(index + 1))
            button.setFixedSize(44, 26)
            button.setToolTip("Color of curve {}. Click to change.".format(index + 1))
            button.clicked.connect(lambda _checked=False, i=index: self._pick_color(i))
            color_row.addWidget(button)
            self._color_buttons.append(button)
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Restore the default curve colors")
        reset_btn.clicked.connect(self._reset_colors)
        color_row.addWidget(reset_btn)
        color_row.addStretch()
        self._update_color_buttons()
        form.addRow("Curve colors:", color_row)

        folder_row = QHBoxLayout()
        self._folder_lbl = QLabel(self._export_folder)
        self._folder_lbl.setWordWrap(True)
        folder_btn = QPushButton("Browse…")
        folder_btn.clicked.connect(self._pick_folder)
        folder_row.addWidget(self._folder_lbl, stretch=1)
        folder_row.addWidget(folder_btn)
        form.addRow("Default export folder:", folder_row)

        # self._language_combo = QComboBox()
        # self._language_combo.addItems(LANGUAGES)
        # saved_language = self._settings.value("language", DEFAULT_LANGUAGE)
        # idx = self._language_combo.findText(saved_language)
        # self._language_combo.setCurrentIndex(idx if idx >= 0 else 0)
        # form.addRow("Language:", self._language_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(model_box)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _build_model_box(self, inference):
        """The "AI model" group: Local (model file) or Server (address)."""
        box = QGroupBox("AI model")
        self._local_radio = QRadioButton("On this computer (Local)")
        self._server_radio = QRadioButton("On a server")
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._local_radio)
        self._mode_group.addButton(self._server_radio)

        self._weights_lbl = QLabel()
        self._weights_lbl.setWordWrap(True)
        self._weights_btn = QPushButton("Browse…")
        self._weights_btn.clicked.connect(self._pick_weights)
        weights_row = QHBoxLayout()
        weights_row.addWidget(self._weights_lbl, stretch=1)
        weights_row.addWidget(self._weights_btn)

        self._api_url_edit = QLineEdit(inference.api_url)
        self._api_url_edit.setMinimumWidth(220)

        form = QFormLayout(box)
        form.addRow(self._local_radio)
        form.addRow("Model file:", weights_row)
        form.addRow(self._server_radio)
        form.addRow("Server address:", self._api_url_edit)

        (self._server_radio if inference.mode == MODE_SERVER else self._local_radio).setChecked(True)
        self._local_radio.toggled.connect(self._update_model_box)
        if FORCED_INFERENCE_MODE:
            # Fixed by this version of the app; see config.py.
            self._local_radio.setEnabled(False)
            self._server_radio.setEnabled(False)
            form.addRow(QLabel("Set by this version of the app."))
        if FORCED_API_URL:
            self._api_url_edit.setReadOnly(True)
        self._update_model_box()
        return box

    def _update_model_box(self):
        local = self._local_radio.isChecked()
        self._weights_lbl.setText(self._weights_path or "No model file chosen")
        self._weights_lbl.setEnabled(local)
        self._weights_btn.setEnabled(local)
        self._api_url_edit.setEnabled(not local)

    def _pick_weights(self):
        start = os.path.dirname(self._weights_path) if self._weights_path else ""
        path = choose_model_file(self, start)
        if path:
            self._weights_path = path
            self._update_model_box()

    def _update_color_buttons(self):
        for button, color in zip(self._color_buttons, self._curve_colors):
            # Dark number on a light color, light number on a dark one.
            text = "#000000" if color.lightness() > 140 else "#ffffff"
            button.setStyleSheet(
                f"background-color: {color.name()}; color: {text}; font-weight: bold; "
                "border: 1px solid #3a4048; border-radius: 3px;"
            )

    def _pick_color(self, index):
        color = QColorDialog.getColor(
            self._curve_colors[index], self, "Color of Curve {}".format(index + 1)
        )
        if color.isValid():
            self._curve_colors[index] = color
            self._update_color_buttons()

    def _reset_colors(self):
        self._curve_colors = [QColor(color) for color in CURVE_COLORS]
        self._update_color_buttons()

    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Default Export Folder", self._export_folder)
        if folder:
            self._export_folder = folder
            self._folder_lbl.setText(folder)

    def _on_save(self):
        if not FORCED_INFERENCE_MODE:
            mode = MODE_LOCAL if self._local_radio.isChecked() else MODE_SERVER
            self._settings.setValue("inference_mode", mode)
        if not FORCED_API_URL:
            # A blank URL falls back to the default, so requests do not all fail.
            api_url = self._api_url_edit.text().strip() or INFERENCE_API_URL
            self._settings.setValue("inference_api_url", api_url)
        if self._weights_path:
            self._settings.setValue("weights_path", self._weights_path)
        for index, color in enumerate(self._curve_colors):
            self._settings.setValue("curve_color_{}".format(index + 1), color.name())
        self._settings.setValue("export_folder", self._export_folder)
        # self._settings.setValue("language", self._language_combo.currentText())
        self.accept()

    @staticmethod
    def get_inference_settings():
        """Where the AI model runs, read without building the dialog.
        FORCED_* values in config.py win over what the user saved."""
        settings = QSettings(ORG_NAME, APP_KEY)
        mode = FORCED_INFERENCE_MODE or settings.value("inference_mode", DEFAULT_INFERENCE_MODE)
        if mode not in (MODE_LOCAL, MODE_SERVER):
            mode = DEFAULT_INFERENCE_MODE
        api_url = FORCED_API_URL or settings.value("inference_api_url", INFERENCE_API_URL)
        # A saved file that has since been moved or deleted is ignored, so a
        # model file placed next to the app can still be found.
        weights_path = settings.value("weights_path", "")
        if not weights_path or not os.path.isfile(weights_path):
            weights_path = find_model_file() or ""
        return InferenceSettings(mode=mode, api_url=api_url, weights_path=weights_path)

    @staticmethod
    def save_weights_path(path):
        """Remember the model file chosen outside the dialog (first run)."""
        QSettings(ORG_NAME, APP_KEY).setValue("weights_path", path)

    @staticmethod
    def get_saved_curve_colors():
        """The saved color of each curve as hex strings, top curve first.
        Anything missing or invalid falls back to the default."""
        settings = QSettings(ORG_NAME, APP_KEY)
        colors = []
        for index, default in enumerate(CURVE_COLORS):
            saved = settings.value("curve_color_{}".format(index + 1))
            if saved is None and index == 0:
                # Older versions had one line color; keep it for curve 1.
                saved = settings.value("line_color")
            color = QColor(saved) if saved else QColor()
            colors.append(color.name() if color.isValid() else default)
        return colors

    @staticmethod
    def get_saved_export_folder():
        settings = QSettings(ORG_NAME, APP_KEY)
        return settings.value("export_folder", os.path.expanduser("~"))

    # @staticmethod
    # def get_saved_language():
    #     """Reads the persisted UI language without constructing the dialog
    #     UI. Defaults to English (the only implemented option so far)."""
    #     settings = QSettings(ORG_NAME, APP_KEY)
    #     return settings.value("language", DEFAULT_LANGUAGE)
