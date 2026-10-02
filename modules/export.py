# Export dialog. Only Raw JSON works today; Annotated Image, PDF Report and
# CSV are shown disabled as "coming soon".

import os
from datetime import datetime

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QDialogButtonBox
)

from modules.utils import export_json_data
from modules.settings_dialog import SettingsDialog


class ExportDialog(QDialog):
    """Choose export formats and a destination folder."""

    def __init__(self, model_engine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setMinimumWidth(380)
        self.model_engine = model_engine

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose what to export:"))

        self.json_check = QCheckBox("Raw JSON (detections, keypoints, angles)")
        self.json_check.setChecked(True)
        layout.addWidget(self.json_check)

        image_check = QCheckBox("Annotated Image (coming soon)")
        image_check.setEnabled(False)
        layout.addWidget(image_check)

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

    def _on_export(self):
        if not self.json_check.isChecked():
            QMessageBox.information(self, "Nothing to Export", "Select at least one format.")
            return

        folder = self.folder_edit.text().strip() or os.path.expanduser("~")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"scoliosis_assessment_{timestamp}.json")
        ok, message = export_json_data(self.model_engine.get_raw_data(), path)

        if ok:
            QMessageBox.information(self, "Export Successful", f"Saved to:\n{path}")
            self.accept()
        else:
            QMessageBox.critical(self, "Export Failed", message)
