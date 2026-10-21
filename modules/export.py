# Export dialog: raw JSON and an annotated JPG (the X-ray with the overlay and
# a results panel). PDF Report and CSV are shown disabled as "coming soon".

import os
import re
from datetime import datetime

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QDialogButtonBox
)

from modules.utils import export_json_data
from modules.report_image import render_annotated_image
from modules.settings_dialog import SettingsDialog


class ExportDialog(QDialog):
    """Choose export formats and a destination folder."""

    # JPEG quality, 0-100.
    JPG_QUALITY = 95

    def __init__(self, model_engine, parent=None, source_name=None, image=None,
                 curve_colors=None, edited=False):
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setMinimumWidth(380)
        self.model_engine = model_engine
        # File name of the X-ray, used in the export file names.
        self.source_name = source_name
        # For the annotated JPG: the X-ray, the curve colors, and whether the
        # landmarks were edited by hand.
        self.image = image
        self.curve_colors = curve_colors
        self.edited = edited

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose what to export:"))

        self.json_check = QCheckBox("Raw JSON (detections, keypoints, angles)")
        self.json_check.setChecked(True)
        layout.addWidget(self.json_check)

        self.image_check = QCheckBox("Annotated Image (JPG, with results panel)")
        has_image = image is not None and not image.isNull()
        self.image_check.setChecked(has_image)
        self.image_check.setEnabled(has_image)
        layout.addWidget(self.image_check)

        pdf_check = QCheckBox("PDF Report (coming soon)")
        pdf_check.setEnabled(False)
        layout.addWidget(pdf_check)

        csv_check = QCheckBox("CSV (coming soon)")
        csv_check.setEnabled(False)
        layout.addWidget(csv_check)

        layout.addWidget(QLabel("Export to folder:"))
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit(SettingsDialog.get_saved_export_folder())
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._pick_folder)
        folder_row.addWidget(self.folder_edit, stretch=1)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        export_btn = buttons.addButton("Export", QDialogButtonBox.AcceptRole)
        export_btn.clicked.connect(self._on_export)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Export Folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)

    def _export_file_name(self, timestamp, extension):
        """scoliosis_assessment_<image name>_<timestamp>.<extension>, so an
        export can be matched to its X-ray. Spaces and characters that are
        not allowed in file names become "_"."""
        stem = os.path.splitext(os.path.basename(self.source_name or ""))[0]
        stem = re.sub(r'[<>:"/\\|?*\s\x00-\x1f]+', "_", stem).strip("._")[:60]
        middle = f"{stem}_" if stem else ""
        return f"scoliosis_assessment_{middle}{timestamp}.{extension}"

    def _on_export(self):
        want_json = self.json_check.isChecked()
        want_image = self.image_check.isChecked()
        if not (want_json or want_image):
            QMessageBox.information(self, "Nothing to Export", "Select at least one format.")
            return

        folder = self.folder_edit.text().strip() or os.path.expanduser("~")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved, errors = [], []

        if want_json:
            path = os.path.join(folder, self._export_file_name(timestamp, "json"))
            ok, message = export_json_data(self.model_engine.get_raw_data(), path)
            (saved if ok else errors).append(path if ok else message)

        if want_image:
            path = os.path.join(folder, self._export_file_name(timestamp, "jpg"))
            try:
                os.makedirs(folder, exist_ok=True)
                rendered = render_annotated_image(
                    self.image, self.model_engine, self.curve_colors, self.source_name, self.edited
                )
                if rendered.save(path, "JPG", self.JPG_QUALITY):
                    saved.append(path)
                else:
                    errors.append(f"Could not write the image: {path}")
            except OSError as exc:
                errors.append(f"Could not write the image: {exc}")

        if errors:
            done = ("Saved:\n" + "\n".join(saved) + "\n\n") if saved else ""
            QMessageBox.critical(self, "Export Failed", done + "\n".join(errors))
            return
        QMessageBox.information(self, "Export Successful", "Saved to:\n" + "\n".join(saved))
        self.accept()
