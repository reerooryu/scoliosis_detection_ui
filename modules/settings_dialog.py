# Settings window: where the AI model runs (this computer or a server),
# curve colors and default export folder. Reading and saving the values is
# done by modules/settings.py.

import os

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QColorDialog, QDialogButtonBox, QRadioButton,
    QButtonGroup, QGroupBox,
    # QComboBox,  # only needed for the language switch below -- re-add on import if re-enabling
)

from config import CURVE_COLORS, FORCED_API_URL, FORCED_INFERENCE_MODE
from modules import settings
from modules.settings import MODE_LOCAL, MODE_SERVER

# --- Language switch (disabled until Thai translations exist) --------------
# To re-enable: uncomment this block, the QComboBox import above, the
# "Language:" row in __init__ and the language line in _on_save, and read it
# back from QSettings key "language" in modules/settings.py.
# LANGUAGES = ["English", "Thai"]
# DEFAULT_LANGUAGE = "English"
# ---------------------------------------------------------------------------

MODEL_FILE_FILTER = "AI model file (*.pth);;All files (*)"


def path_field(placeholder=""):
    """A one-line, read-only box for a file or folder path (changed with a
    Browse button). A long path scrolls inside the box instead of being cut
    off; the full path is also shown when the mouse rests on it."""
    field = QLineEdit()
    field.setReadOnly(True)
    field.setPlaceholderText(placeholder)
    field.setMinimumWidth(300)
    return field


def set_path(field, path):
    """Show path in a path_field(). The end of a long path (the file or
    folder name) is the part left in view."""
    field.setText(path)
    field.setToolTip(path)
    field.setCursorPosition(len(path))


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
        self.setMinimumWidth(560)

        self._curve_colors = [QColor(color) for color in settings.curve_colors()]
        self._export_folder = settings.export_folder()
        inference = settings.inference_settings()
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
        self._folder_field = path_field()
        set_path(self._folder_field, self._export_folder)
        folder_btn = QPushButton("Browse…")
        folder_btn.clicked.connect(self._pick_folder)
        folder_row.addWidget(self._folder_field, stretch=1)
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

        self._weights_field = path_field("No model file chosen")
        self._weights_btn = QPushButton("Browse…")
        self._weights_btn.clicked.connect(self._pick_weights)
        weights_row = QHBoxLayout()
        weights_row.addWidget(self._weights_field, stretch=1)
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
        set_path(self._weights_field, self._weights_path)
        self._weights_field.setEnabled(local)
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
            set_path(self._folder_field, folder)

    def _on_save(self):
        settings.save_inference_settings(
            MODE_LOCAL if self._local_radio.isChecked() else MODE_SERVER,
            self._api_url_edit.text(),
            self._weights_path,
        )
        settings.save_curve_colors([color.name() for color in self._curve_colors])
        settings.save_export_folder(self._export_folder)
        # QSettings(...).setValue("language", self._language_combo.currentText())
        self.accept()
