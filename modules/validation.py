# Model validation / QA comparison dialog.
#
# Lets the ML team compare a model prediction JSON (e.g. a saved API
# response) against a ground-truth label JSON using the same
# detections/keypoints/angle_pairs schema, and reports the checks outlined
# by the team: vertebra count match, per-vertebra oblique-angle accuracy,
# Cobb curve count match, Cobb angle accuracy, and a note on processing
# time.
#
# This is a separate workflow from the clinical assessment in
# modules/main_window.py -- it needs a ground-truth label a real patient
# assessment never has, so it's its own dialog rather than folded into the
# clinical canvas/panel. Native OS dialog styling (no clinical theme),
# consistent with every other dialog in this app.

import json
import os

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QTextEdit
)

from modules.geometry import (
    compare_detection_counts, compare_oblique_angles,
    compare_cobb_angle_counts, compare_cobb_angle_values
)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


class ValidationDialog(QDialog):
    """Compare a model prediction against a ground-truth label file."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Validation")
        self.setMinimumSize(600, 540)

        self.prediction_data = None
        self.label_data = None

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Compare a model prediction against a ground-truth label file "
            "(both using the detections/keypoints/angle_pairs JSON schema)."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        pred_row = QHBoxLayout()
        self.pred_path_lbl = QLabel("No prediction file loaded")
        pred_btn = QPushButton("Load Prediction JSON…")
        pred_btn.clicked.connect(self._load_prediction)
        pred_row.addWidget(self.pred_path_lbl, stretch=1)
        pred_row.addWidget(pred_btn)
        layout.addLayout(pred_row)

        label_row = QHBoxLayout()
        self.label_path_lbl = QLabel("No ground-truth label file loaded")
        label_btn = QPushButton("Load Ground-Truth Label JSON…")
        label_btn.clicked.connect(self._load_label)
        label_row.addWidget(self.label_path_lbl, stretch=1)
        label_row.addWidget(label_btn)
        layout.addLayout(label_row)

        compare_btn = QPushButton("Compare")
        compare_btn.clicked.connect(self._on_compare)
        layout.addWidget(compare_btn)

        self.results_txt = QTextEdit()
        self.results_txt.setReadOnly(True)
        self.results_txt.setFont(QFont("Consolas", 11))
        layout.addWidget(self.results_txt, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn)

    def _load_prediction(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Prediction JSON", "", "JSON Files (*.json)")
        if path:
            try:
                self.prediction_data = _load_json(path)
                self.pred_path_lbl.setText(os.path.basename(path))
            except Exception as exc:
                QMessageBox.critical(self, "Load Failed", str(exc))

    def _load_label(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Ground-Truth Label JSON", "", "JSON Files (*.json)")
        if path:
            try:
                self.label_data = _load_json(path)
                self.label_path_lbl.setText(os.path.basename(path))
            except Exception as exc:
                QMessageBox.critical(self, "Load Failed", str(exc))

    def _on_compare(self):
        if self.prediction_data is None or self.label_data is None:
            QMessageBox.information(
                self, "Missing Files",
                "Load both a prediction and a ground-truth label file first."
            )
            return
        self.results_txt.setPlainText(self._build_report())

    def _build_report(self):
        pred_dets = self.prediction_data.get("detections", [])
        label_dets = self.label_data.get("detections", [])
        pred_pairs = self.prediction_data.get("angle_pairs", [])
        label_pairs = self.label_data.get("angle_pairs", [])

        count_cmp = compare_detection_counts(pred_dets, label_dets)
        angle_cmp = compare_oblique_angles(pred_dets, label_dets)
        cobb_count_cmp = compare_cobb_angle_counts(pred_pairs, label_pairs)
        cobb_val_cmp = compare_cobb_angle_values(pred_pairs, label_pairs)

        lines = []
        lines.append("=" * 64)
        lines.append("1. SEGMENTATION")
        lines.append("=" * 64)
        lines.append(f"1.1 Vertebra count -- predicted: {count_cmp['predicted']}, label: {count_cmp['label']}")
        lines.append(f"    Verdict: {count_cmp['verdict']}")
        lines.append("")
        lines.append(f"1.2 Per-vertebra oblique angle error ({len(angle_cmp['rows'])} matched by index)")
        lines.append(f"    Mean upper-oblique error: {angle_cmp['mean_upper_error']:.2f} deg")
        lines.append(f"    Mean lower-oblique error: {angle_cmp['mean_lower_error']:.2f} deg")
        lines.append(f"    Max  upper-oblique error: {angle_cmp['max_upper_error']:.2f} deg")
        lines.append(f"    Max  lower-oblique error: {angle_cmp['max_lower_error']:.2f} deg")
        lines.append("")
        lines.append("=" * 64)
        lines.append("2. REGIONAL MEASUREMENTS")
        lines.append("=" * 64)
        lines.append(f"2.1 Cobb curve count -- predicted: {cobb_count_cmp['predicted']}, label: {cobb_count_cmp['label']}")
        lines.append(f"    Verdict: {cobb_count_cmp['verdict']}")
        lines.append("")
        lines.append("    Per-curve Cobb angle accuracy:")
        for row in cobb_val_cmp["rows"]:
            lines.append(
                f"      Curve #{row['pair_index'] + 1}: predicted {row['predicted']:.2f} deg, "
                f"label {row['label']:.2f} deg, error {row['error']:.2f} deg"
            )
        if not cobb_val_cmp["rows"]:
            lines.append("      (no overlapping curves to compare)")
        lines.append(f"    Mean Cobb angle error: {cobb_val_cmp['mean_error']:.2f} deg")
        lines.append(f"    Max  Cobb angle error: {cobb_val_cmp['max_error']:.2f} deg")
        lines.append("")
        lines.append("    2.2 CSVL is shown in the clinical workspace panel, not compared")
        lines.append("        here -- the label schema captured so far has no CSVL/apex field.")
        lines.append("")
        lines.append("=" * 64)
        lines.append("3. ANALYSIS TIME")
        lines.append("=" * 64)
        lines.append("Reported per-run in the clinical workspace status bar. Since speed is")
        lines.append("device-dependent, always note the hardware alongside any timing figure")
        lines.append("you report to others.")

        return "\n".join(lines)
