# Settings dialog: inference API URL, default overlay line color, and
# default export folder. Preferences are persisted with QSettings so they
# survive app restarts, and are read by modules/main_window.py (API URL),
# modules/overlay.py (line color, once wired in), and the Export modal
# (export folder).

import os

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QColorDialog, QDialogButtonBox
    # QComboBox,  # only needed for the language switch below -- re-add on import if re-enabling
)

from config import INFERENCE_API_URL

ORG_NAME = "ScoliosisSuite"
APP_KEY = "DetectionUI"
DEFAULT_LINE_COLOR = "#ff5722"

# --- Language switch: disabled until Thai translations actually exist ------
# Commented out wholesale rather than left as a disabled dropdown, at the
# user's request, since Thai isn't implemented yet. To re-enable once real
# translations (Qt Linguist .ts/.qm or a string table) are wired up:
#   1. Uncomment this block and the QComboBox import above.
#   2. Uncomment the "Language:" form row in __init__.
#   3. Uncomment the language line in _on_save.
#   4. Uncomment get_saved_language() below.
# LANGUAGES = ["English", "Thai"]
# DEFAULT_LANGUAGE = "English"
# ---------------------------------------------------------------------------


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(380)

        self._settings = QSettings(ORG_NAME, APP_KEY)
        self._line_color = QColor(self._settings.value("line_color", DEFAULT_LINE_COLOR))
        self._export_folder = self._settings.value("export_folder", os.path.expanduser("~"))

        form = QFormLayout()

        self._api_url_edit = QLineEdit(self._settings.value("inference_api_url", INFERENCE_API_URL))
        self._api_url_edit.setMinimumWidth(220)
        form.addRow("Inference API URL:", self._api_url_edit)

        color_row = QHBoxLayout()
        self._color_swatch = QLabel()
        self._color_swatch.setFixedSize(24, 24)
        self._update_swatch()
        color_btn = QPushButton("Choose…")
        color_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self._color_swatch)
        color_row.addWidget(color_btn)
        color_row.addStretch()
        form.addRow("Default line color:", color_row)

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

    def _update_swatch(self):
        self._color_swatch.setStyleSheet(
            f"background-color: {self._line_color.name()}; "
            "border: 1px solid #3a4048; border-radius: 3px;"
        )

    def _pick_color(self):
        color = QColorDialog.getColor(self._line_color, self, "Default Line Color")
        if color.isValid():
            self._line_color = color
            self._update_swatch()

    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Default Export Folder", self._export_folder)
        if folder:
            self._export_folder = folder
            self._folder_lbl.setText(folder)

    def _on_save(self):
        self._settings.setValue("inference_api_url", self._api_url_edit.text().strip())
        self._settings.setValue("line_color", self._line_color.name())
        self._settings.setValue("export_folder", self._export_folder)
        # self._settings.setValue("language", self._language_combo.currentText())
        self.accept()

    @staticmethod
    def get_saved_api_url():
        """Reads the persisted API URL without constructing the dialog UI."""
        settings = QSettings(ORG_NAME, APP_KEY)
        return settings.value("inference_api_url", INFERENCE_API_URL)

    @staticmethod
    def get_saved_line_color():
        """Returns the persisted Cobb measurement-line color as a valid hex value."""
        settings = QSettings(ORG_NAME, APP_KEY)
        color = QColor(settings.value("line_color", DEFAULT_LINE_COLOR))
        return color.name() if color.isValid() else DEFAULT_LINE_COLOR

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
