# Load page: image import (Open Image + Display preview + Submit transition).
#
# Handles drag-and-drop / file-browse import of a spine X-ray image, validates
# the file type, and shows a thumbnail + Submit control. This module never
# talks to the backend/model -- it only stores the selected image path and
# hands it off via the `submitted` signal.

import os
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QPixmap, QFont, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QFileDialog, QMessageBox, QSizePolicy
)

from config import SUPPORTED_IMAGE_EXTENSIONS
from modules.theme import ACCENT


def _is_supported_image(path):
    return os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS


def _build_upload_icon(size=56, color=ACCENT):
    """Draws a simple upload-arrow-into-tray glyph with QPainter.

    Used instead of an emoji character: emoji glyph coverage/rendering
    varies a lot across OS/font installs (it showed up as a blank "tofu"
    box in headless testing here), which isn't something clinical software
    should depend on for a UI it's actually used every day.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(size * 0.06)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)

    cx = size / 2
    top = size * 0.12
    shaft_bottom = size * 0.60
    arrow_w = size * 0.22

    painter.drawLine(QPointF(cx, shaft_bottom), QPointF(cx, top))
    painter.drawLine(QPointF(cx, top), QPointF(cx - arrow_w, top + arrow_w))
    painter.drawLine(QPointF(cx, top), QPointF(cx + arrow_w, top + arrow_w))

    tray_y = size * 0.82
    painter.drawLine(QPointF(size * 0.18, size * 0.70), QPointF(size * 0.18, tray_y))
    painter.drawLine(QPointF(size * 0.82, size * 0.70), QPointF(size * 0.82, tray_y))
    painter.drawLine(QPointF(size * 0.18, tray_y), QPointF(size * 0.82, tray_y))

    painter.end()
    return pixmap


class DropZone(QFrame):
    """Drag-and-drop landing area with a Browse fallback."""
    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setProperty("dragActive", False)
        self.setAcceptDrops(True)
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)

        icon = QLabel()
        icon.setPixmap(_build_upload_icon(56))
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        text = QLabel("Drag & drop a spine X-ray image here\n— or —")
        text.setAlignment(Qt.AlignCenter)
        layout.addWidget(text)

        browse_btn = QPushButton("Browse Files…")
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(browse_btn, alignment=Qt.AlignCenter)

        hint = QLabel("Supported formats: JPG, JPEG, PNG")
        hint.setObjectName("MetricLabel")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

    def _on_browse(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Spine X-Ray", "", "Image Files (*.jpg *.jpeg *.png)"
        )
        if file_path:
            self._validate_and_emit(file_path)

    def _validate_and_emit(self, file_path):
        if not _is_supported_image(file_path):
            QMessageBox.warning(
                self, "Unsupported File",
                "Please select a JPG, JPEG, or PNG image file."
            )
            return
        self.file_dropped.emit(file_path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if _is_supported_image(path):
                self.setProperty("dragActive", True)
                self.style().unpolish(self)
                self.style().polish(self)
                event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        urls = event.mimeData().urls()
        if urls:
            self._validate_and_emit(urls[0].toLocalFile())


class LoadPage(QWidget):
    """Full import page: drop zone + thumbnail preview + Submit action."""
    submitted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Plain QWidget subclasses don't paint their stylesheet background
        # when embedded as a non-top-level child (only real windows and
        # style-aware widgets like QFrame/QPushButton do that automatically).
        # Without this, LoadPage renders with the native OS window color
        # once it's nested inside the QStackedWidget/QMainWindow, even
        # though the same stylesheet renders fine as a standalone window.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.image_path = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 40)
        layout.setSpacing(20)

        title = QLabel("Load Spine X-Ray")
        title.setFont(QFont("Segoe UI", 17, QFont.Bold))
        layout.addWidget(title)

        subtitle = QLabel("Import a spine X-ray image to begin a Cobb angle assessment.")
        subtitle.setObjectName("MetricLabel")
        layout.addWidget(subtitle)

        self.drop_zone = DropZone(self)
        self.drop_zone.file_dropped.connect(self._on_file_selected)
        layout.addWidget(self.drop_zone, stretch=1)

        # Thumbnail preview row (hidden until a valid image is loaded)
        self.preview_frame = QFrame(self)
        self.preview_frame.setVisible(False)
        preview_layout = QHBoxLayout(self.preview_frame)

        self.thumb_lbl = QLabel(self)
        self.thumb_lbl.setFixedSize(90, 120)
        self.thumb_lbl.setScaledContents(True)
        preview_layout.addWidget(self.thumb_lbl)

        info_layout = QVBoxLayout()
        self.file_name_lbl = QLabel(self)
        self.file_name_lbl.setObjectName("MetricValue")
        self.file_meta_lbl = QLabel(self)
        self.file_meta_lbl.setObjectName("MetricLabel")
        info_layout.addWidget(self.file_name_lbl)
        info_layout.addWidget(self.file_meta_lbl)
        preview_layout.addLayout(info_layout)
        preview_layout.addStretch()

        layout.addWidget(self.preview_frame)

        # Submit action
        action_row = QHBoxLayout()
        action_row.addStretch()
        self.submit_btn = QPushButton("Submit →")
        self.submit_btn.setObjectName("PrimaryButton")
        self.submit_btn.setEnabled(False)
        self.submit_btn.setCursor(Qt.PointingHandCursor)
        self.submit_btn.clicked.connect(self._on_submit)
        action_row.addWidget(self.submit_btn)
        layout.addLayout(action_row)

    def _on_file_selected(self, file_path):
        """Validates, previews, and stores the image path internally."""
        if not os.path.exists(file_path):
            return
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.critical(self, "Error", "Invalid or corrupted image file.")
            return

        self.image_path = file_path

        scaled = pixmap.scaled(90, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.thumb_lbl.setPixmap(scaled)
        self.thumb_lbl.setFixedSize(scaled.size())
        self.file_name_lbl.setText(os.path.basename(file_path))
        size_kb = os.path.getsize(file_path) / 1024.0
        self.file_meta_lbl.setText(
            f"{pixmap.width()}×{pixmap.height()} px  •  {size_kb:.1f} KB"
        )
        self.preview_frame.setVisible(True)
        self.submit_btn.setEnabled(True)

    def reset(self):
        """Clears the loaded image so the page can be reused (e.g. via Reset)."""
        self.image_path = None
        self.preview_frame.setVisible(False)
        self.submit_btn.setEnabled(False)

    def _on_submit(self):
        if self.image_path:
            self.submitted.emit(self.image_path)
