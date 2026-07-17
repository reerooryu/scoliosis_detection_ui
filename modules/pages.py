# Qt Wizard Pages for Scoliosis Detection UI

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QIcon, QFont
from PySide6.QtWidgets import (
    QWizardPage, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QFrame, QTextEdit, QMessageBox, QSplitter, QProgressBar, QSizePolicy,
    QLineEdit
)
import os
from config import APP_NAME
from modules.canvas import ScoliosisInteractiveCanvas
from modules.model_mock import ScoliosisModelEngine
from modules.utils import generate_clinical_summary, export_json_data

class DragDropArea(QFrame):
    """Custom Drag & Drop panel representing a prominent landing area for image files."""
    file_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DragDropArea")
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(250)

        # Style the drag area using simple native border cues
        self.setStyleSheet("""
            QFrame#DragDropArea {
                border: 2px dashed #888888;
                border-radius: 8px;
                background-color: #fafafa;
            }
            QFrame#DragDropArea:hover {
                border-color: #0078d7;
                background-color: #f0f7ff;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # Label indicators
        self.icon_label = QLabel(self)
        self.icon_label.setText("📁")
        self.icon_label.setFont(QFont("Arial", 48))
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)

        self.text_label = QLabel("Drag & Drop your Spine X-Ray Image here\n- or -", self)
        self.text_label.setFont(QFont("Segoe UI", 11))
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet("color: #555555;")
        layout.addWidget(self.text_label)

        self.browse_button = QPushButton("Browse Image...", self)
        self.browse_button.setFont(QFont("Segoe UI", 10))
        self.browse_button.setCursor(Qt.PointingHandCursor)
        self.browse_button.clicked.connect(self.on_browse_clicked)
        layout.addWidget(self.browse_button)

    def on_browse_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Spine X-Ray", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.file_selected.emit(file_path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) > 0 and urls[0].toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                event.acceptProposedAction()
                self.setStyleSheet("""
                    QFrame#DragDropArea {
                        border: 2px dashed #0078d7;
                        background-color: #e6f2ff;
                    }
                """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame#DragDropArea {
                border: 2px dashed #888888;
                background-color: #fafafa;
            }
        """)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.file_selected.emit(file_path)


# ==========================================
# PAGE 1: Image Loader
# ==========================================
class ImageLoaderPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Step 1: Load X-Ray Image")
        self.setSubTitle("Select or drag and drop a spine X-Ray image to begin analysis.")
        
        # Hidden QLineEdit to register as a mandatory field for validation
        self.image_path_field = QLineEdit("", self)
        self.image_path_field.setVisible(False)
        self.registerField("image_path*", self.image_path_field)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.addWidget(self.image_path_field)

        # Drag and Drop landing area
        self.drag_drop_area = DragDropArea(self)
        self.drag_drop_area.file_selected.connect(self.load_image_path)
        layout.addWidget(self.drag_drop_area)

        # Thumbnail Preview Panel
        self.preview_frame = QFrame(self)
        self.preview_frame.setFrameStyle(QFrame.StyledPanel)
        self.preview_frame.setVisible(False)
        preview_layout = QHBoxLayout(self.preview_frame)

        self.thumbnail_lbl = QLabel(self)
        self.thumbnail_lbl.setFixedSize(120, 160)
        self.thumbnail_lbl.setScaledContents(True)
        self.thumbnail_lbl.setFrameStyle(QFrame.Box | QFrame.Plain)
        preview_layout.addWidget(self.thumbnail_lbl)

        info_layout = QVBoxLayout()
        info_layout.setAlignment(Qt.AlignVCenter)
        self.file_name_lbl = QLabel(self)
        self.file_name_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.file_size_lbl = QLabel(self)
        info_layout.addWidget(self.file_name_lbl)
        info_layout.addWidget(self.file_size_lbl)
        
        self.change_img_btn = QPushButton("Select Different Image...", self)
        self.change_img_btn.clicked.connect(self.on_change_clicked)
        info_layout.addWidget(self.change_img_btn)
        
        preview_layout.addLayout(info_layout)
        preview_layout.addStretch()
        layout.addWidget(self.preview_frame)

    def load_image_path(self, file_path):
        """Validates, displays the thumbnail, and saves to the wizard state."""
        if not os.path.exists(file_path):
            return

        # Update wizard state variable
        if self.wizard():
            self.wizard().image_path = file_path
        self.image_path_field.setText(file_path)

        # Update UI thumbnail preview
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.critical(self, "Error", "Invalid or corrupted image file.")
            return

        # Scale to fit while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(
            150, 200, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.thumbnail_lbl.setPixmap(scaled_pixmap)
        self.thumbnail_lbl.setFixedSize(scaled_pixmap.size())
        
        self.file_name_lbl.setText(f"Loaded: {os.path.basename(file_path)}")
        
        file_size_kb = os.path.getsize(file_path) / 1024.0
        self.file_size_lbl.setText(f"Resolution: {pixmap.width()}x{pixmap.height()} | Size: {file_size_kb:.1f} KB")

        # Swap visibility of loader and preview
        self.drag_drop_area.setVisible(False)
        self.preview_frame.setVisible(True)

        # Complete the page to enable Next button
        self.completeChanged.emit()

    def on_change_clicked(self):
        """Resets loader state to pick a different image."""
        if self.wizard():
            self.wizard().image_path = ""
        self.image_path_field.clear()
        self.preview_frame.setVisible(False)
        self.drag_drop_area.setVisible(True)
        self.completeChanged.emit()

    def isComplete(self):
        """Ensures next is enabled only when image path is valid."""
        img_path = self.image_path_field.text()
        return bool(img_path and os.path.exists(img_path))


# ==========================================
# PAGE 2: Model Inference & Visualization
# ==========================================
class InferencePage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Step 2: Model Inference & Visualization")
        self.setSubTitle("Deep learning segmentation model results overlayed on the image.")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Splitter to hold Canvas (left) and Metrics panel (right)
        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.splitter)

        # Left: Canvas area
        canvas_container = QFrame(self.splitter)
        canvas_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = ScoliosisInteractiveCanvas(canvas_container)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_layout.addWidget(self.canvas)
        
        # Simple zoom toolbar
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addStretch()
        self.zoom_in_btn = QPushButton("Zoom In (+)", self)
        self.zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        self.zoom_out_btn = QPushButton("Zoom Out (-)", self)
        self.zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        self.fit_btn = QPushButton("Fit To View", self)
        self.fit_btn.clicked.connect(self.canvas.fit_in_view)
        
        toolbar_layout.addWidget(self.zoom_out_btn)
        toolbar_layout.addWidget(self.zoom_in_btn)
        toolbar_layout.addWidget(self.fit_btn)
        canvas_layout.addLayout(toolbar_layout)

        self.splitter.addWidget(canvas_container)

        # Right: Info & Initial Results Summary panel
        info_panel = QFrame(self.splitter)
        info_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        info_layout = QVBoxLayout(info_panel)
        
        info_lbl = QLabel("Initial AI Analysis", info_panel)
        info_lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
        info_layout.addWidget(info_lbl)

        self.summary_txt = QTextEdit(info_panel)
        self.summary_txt.setReadOnly(True)
        self.summary_txt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.summary_txt.setFont(QFont("Consolas", 10))
        info_layout.addWidget(self.summary_txt)
        
        self.splitter.addWidget(info_panel)
        
        # Set default proportions: 70% Canvas, 30% Info
        self.splitter.setSizes([750, 300])

    def initializePage(self):
        """Runs mock deep learning model and visualizes initial overlays."""
        # Retrieve image path from shared state
        image_path = self.wizard().image_path
        pixmap = QPixmap(image_path)
        
        # Reset and scale model engine coordinates to match active loaded image resolution
        model_engine = self.wizard().model_engine
        model_engine.load_data()  # Reset to raw coordinates
        model_engine.scale_coordinates(pixmap.width(), pixmap.height())
        
        self.canvas.load_image(pixmap)
        
        # Render non-interactive features (view-only)
        self.canvas.set_interactive(False)
        self.canvas.render_model_data(model_engine)
        
        # Display initial report
        summary_text = generate_clinical_summary(model_engine.get_raw_data())
        self.summary_txt.setPlainText(summary_text)


# ==========================================
# PAGE 3: Interactive Adjustment
# ==========================================
class AdjustmentPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Step 3: Interactive Adjustment")
        self.setSubTitle("Click and drag any vertebra center or corner point to manually refine coordinates.")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Splitter to hold Interactive Canvas (left) and live coordinates/angles (right)
        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.splitter)

        # Left: Interactive Canvas
        canvas_container = QFrame(self.splitter)
        canvas_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = ScoliosisInteractiveCanvas(canvas_container)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Connect dragging signals to trigger real-time updates
        self.canvas.signals.keypoint_moved.connect(self.on_keypoint_dragged)
        canvas_layout.addWidget(self.canvas)
        
        # Zoom toolbar
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addStretch()
        self.zoom_in_btn = QPushButton("Zoom In (+)", self)
        self.zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        self.zoom_out_btn = QPushButton("Zoom Out (-)", self)
        self.zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        self.fit_btn = QPushButton("Fit To View", self)
        self.fit_btn.clicked.connect(self.canvas.fit_in_view)
        
        toolbar_layout.addWidget(self.zoom_out_btn)
        toolbar_layout.addWidget(self.zoom_in_btn)
        toolbar_layout.addWidget(self.fit_btn)
        canvas_layout.addLayout(toolbar_layout)

        self.splitter.addWidget(canvas_container)

        # Right: Dynamic Clinical metrics panels
        info_panel = QFrame(self.splitter)
        info_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        info_layout = QVBoxLayout(info_panel)
        
        lbl = QLabel("Live Cobb Angle Metrics", info_panel)
        lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
        info_layout.addWidget(lbl)

        self.live_metrics_txt = QTextEdit(info_panel)
        self.live_metrics_txt.setReadOnly(True)
        self.live_metrics_txt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.live_metrics_txt.setFont(QFont("Consolas", 10))
        info_layout.addWidget(self.live_metrics_txt)
        
        self.splitter.addWidget(info_panel)
        self.splitter.setSizes([750, 300])

    def initializePage(self):
        """Enables interactive keypoint dragging."""
        image_path = self.wizard().image_path
        pixmap = QPixmap(image_path)
        self.canvas.load_image(pixmap)
        
        model_engine = self.wizard().model_engine
        
        # Render interactive features (draggable keypoints)
        self.canvas.set_interactive(True)
        self.canvas.render_model_data(model_engine)
        
        # Display live metrics summary
        self.update_live_metrics_panel()

    def on_keypoint_dragged(self, det_idx, kp_idx, new_x, new_y):
        """Handles real-time drag-and-drop feedback."""
        model_engine = self.wizard().model_engine
        
        # Update coordinate in the clinical calculation engine
        model_engine.update_keypoint(det_idx, kp_idx, new_x, new_y)
        
        # Re-render canvas lines & text boxes instantly (no flickering)
        self.canvas.render_model_data(model_engine)
        
        # Update the sidebar clinical report
        self.update_live_metrics_panel()

    def update_live_metrics_panel(self):
        model_engine = self.wizard().model_engine
        summary_text = generate_clinical_summary(model_engine.get_raw_data())
        self.live_metrics_txt.setPlainText(summary_text)


# ==========================================
# PAGE 4: Export Data
# ==========================================
class ExportPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Step 4: Clinical Export")
        self.setSubTitle("Review finalized clinical report and export results to structured JSON.")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        title_lbl = QLabel("Finalized Scoliosis Analysis Report Summary", self)
        title_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout.addWidget(title_lbl)

        # Final Summary text viewer
        self.final_report_txt = QTextEdit(self)
        self.final_report_txt.setReadOnly(True)
        self.final_report_txt.setFont(QFont("Consolas", 10))
        layout.addWidget(self.final_report_txt)

        # Actions Layout
        action_layout = QHBoxLayout()
        self.export_btn = QPushButton("📁 Export JSON...", self)
        self.export_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 10px 24px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        self.export_btn.setCursor(Qt.PointingHandCursor)
        self.export_btn.clicked.connect(self.on_export_clicked)
        
        action_layout.addStretch()
        action_layout.addWidget(self.export_btn)
        action_layout.addStretch()
        
        layout.addLayout(action_layout)

    def initializePage(self):
        """Locks in the final adjustments and displays the summary report."""
        model_engine = self.wizard().model_engine
        report_text = generate_clinical_summary(model_engine.get_raw_data())
        self.final_report_txt.setPlainText(report_text)

    def on_export_clicked(self):
        """Asks the user where to save the structured JSON output."""
        default_name = "scoliosis_assessment_results.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Final Clinical Export", default_name, "JSON Files (*.json)"
        )
        
        if file_path:
            model_engine = self.wizard().model_engine
            success, message = export_json_data(model_engine.get_raw_data(), file_path)
            
            if success:
                QMessageBox.information(
                    self, "Export Successful", 
                    f"Clinical coordinates and Cobb angles have been successfully exported to:\n{file_path}"
                )
            else:
                QMessageBox.critical(self, "Export Failed", message)
