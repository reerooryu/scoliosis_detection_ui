# Settings dialog: inference API URL, curve colors and default export
# folder. Saved with QSettings, so they survive restarts.

import os

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QColorDialog, QDialogButtonBox
    # QComboBox,  # only needed for the language switch below -- re-add on import if re-enabling
)

from config import INFERENCE_API_URL, CURVE_COLORS

ORG_NAME = "ScoliosisSuite"
APP_KEY = "DetectionUI"

# --- Language switch (disabled until Thai translations exist) --------------
# To re-enable: uncomment this block, the QComboBox import above, the
# "Language:" row in __init__, the language line in _on_save, and
# get_saved_language() below.
# LANGUAGES = ["English", "Thai"]
# DEFAULT_LANGUAGE = "English"
# ---------------------------------------------------------------------------


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(380)

        self._settings = QSettings(ORG_NAME, APP_KEY)
        self._curve_colors = [QColor(color) for color in self.get_saved_curve_colors()]
        self._export_folder = self._settings.value("export_folder", os.path.expanduser("~"))

        form = QFormLayout()

        self._api_url_edit = QLineEdit(self._settings.value("inference_api_url", INFERENCE_API_URL))
        self._api_url_edit.setMinimumWidth(220)
        form.addRow("Inference API URL:", self._api_url_edit)

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
        layout.addLayout(form)
        layout.addWidget(buttons)

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
        # A blank URL falls back to the default, so requests do not all fail.
        api_url = self._api_url_edit.text().strip() or INFERENCE_API_URL
        self._settings.setValue("inference_api_url", api_url)
        for index, color in enumerate(self._curve_colors):
            self._settings.setValue("curve_color_{}".format(index + 1), color.name())
        self._settings.setValue("export_folder", self._export_folder)
        # self._settings.setValue("language", self._language_combo.currentText())
        self.accept()

    @staticmethod
    def get_saved_api_url():
        """Reads the persisted API URL without constructing the dialog UI."""
        settings = QSettings(ORG_NAME, APP_KEY)
        return settings.value("inference_api_url", INFERENCE_API_URL)

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
