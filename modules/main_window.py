# Main application window: single-page clinical workspace.
#
# Owns the toolbar (custom clinical theme) and the native menu bar
# (File/Edit/View/Tools), and swaps between the Load page (image import)
# and the Workspace page (canvas + measurement panel) via a QStackedWidget.
#
# On Submit, this window kicks off a live call to the AI inference backend
# (modules/parser.py) on a background thread, then hands the result to
# ScoliosisModelEngine (modules/model_mock.py) and OverlayLayer
# (modules/overlay.py) to render landmarks/Cobb lines/CSVL and populate the
# measurement panel. If the backend is unreachable, the image still shows
# and the user can retry once it's back.
#
# Tools > Model Validation opens a separate, unrelated workflow
# (modules/validation.py) for comparing a model prediction against a
# ground-truth label file -- that's an ML-team QA task, not something a
# clinician does per-patient, so it's deliberately kept out of this
# window's own state.

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFrame,
    QLabel, QPushButton, QStackedWidget, QToolBar, QSizePolicy, QStatusBar,
    QMessageBox, QDialog
)

from config import APP_NAME, WINDOW_WIDTH, WINDOW_HEIGHT, INFERENCE_TIMEOUT
from modules import theme
from modules.load_view import LoadPage
from modules.canvas import ImageCanvas
from modules.overlay import OverlayLayer
from modules.model_mock import ScoliosisModelEngine
from modules.parser import InferenceWorker
from modules.settings_dialog import SettingsDialog
from modules.export import ExportDialog
from modules.validation import ValidationDialog


class MetricRow(QFrame):
    """A single labeled measurement value in the right-hand panel."""

    def __init__(self, label, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(2)

        lbl = QLabel(label)
        lbl.setObjectName("MetricLabel")
        self.value_lbl = QLabel("—")
        self.value_lbl.setObjectName("MetricValue")

        layout.addWidget(lbl)
        layout.addWidget(self.value_lbl)

    def set_value(self, text):
        self.value_lbl.setText(text)

    def set_tooltip(self, text):
        self.value_lbl.setToolTip(text)


class WorkspacePage(QWidget):
    """Canvas (left) + measurement summary and export action (right)."""

    METRIC_KEYS = [
        "Primary Cobb Angle", "Curve 1", "Curve 2", "Apex",
        "CSVL Deviation", "Vertebrae", "Processing Time"
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        # See LoadPage.__init__ for why this is needed: a plain QWidget
        # embedded as a non-top-level child won't paint its stylesheet
        # background on its own.
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal, self)
        layout.addWidget(splitter)

        self.canvas = ImageCanvas(self)
        splitter.addWidget(self.canvas)

        panel = QFrame(self)
        panel.setObjectName("MeasurementPanel")
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(340)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(18, 18, 18, 18)

        title = QLabel("Measurements")
        title.setObjectName("MetricValue")
        panel_layout.addWidget(title)

        self.metrics = {}
        for key in self.METRIC_KEYS:
            row = MetricRow(key, panel)
            self.metrics[key] = row
            panel_layout.addWidget(row)

        self.metrics["Apex"].set_tooltip(
            "Vertebra with the largest horizontal deviation from the CSVL "
            "(approximated -- see CSVL Deviation tooltip)."
        )
        self.metrics["CSVL Deviation"].set_tooltip(
            "CSVL (Central Sacral Vertical Line) reference is approximated as "
            "the bottommost detected vertebra -- the backend JSON doesn't "
            "label a sacrum. Deviation is in pixels: the JSON has no "
            "pixel-spacing/calibration field to convert to mm."
        )

        panel_layout.addStretch()

        self.export_btn = QPushButton("Export…")
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip("Available once AI analysis results are loaded")
        panel_layout.addWidget(self.export_btn)

        # Full workflow reset lives here, below Export, rather than in the
        # toolbar -- the toolbar instead has "Reset Edits" (undo manual
        # keypoint adjustments only, keeping the loaded image/analysis).
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setToolTip("Clear the loaded image and analysis, and return to the start")
        panel_layout.addWidget(self.reset_btn)

        splitter.addWidget(panel)
        splitter.setSizes([1000, 300])

    def reset_metrics(self):
        for row in self.metrics.values():
            row.set_value("—")


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.image_path = None
        self.model_engine = None
        self._inference_worker = None
        self._dirty = False  # True once landmarks have been dragged since the last export

        self._build_menu_and_toolbar()

        self.stack = QStackedWidget(self)
        self.load_page = LoadPage(self)
        self.workspace_page = WorkspacePage(self)
        self.stack.addWidget(self.load_page)
        self.stack.addWidget(self.workspace_page)
        self.setCentralWidget(self.stack)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Load a spine X-ray image to begin.")

        # Custom clinical theme is applied only to these specific content
        # widgets -- the menu bar and every dialog stay native OS style.
        theme.apply_clinical_theme(self.toolbar, self.stack, self.statusBar())

        self.overlay_layer = OverlayLayer(self.workspace_page.canvas)
        self.overlay_layer.signals.keypoint_moved.connect(self._on_keypoint_dragged)
        self.overlay_layer.signals.drag_started.connect(self._on_drag_started)
        self.overlay_layer.signals.drag_finished.connect(self._on_drag_finished)

        self.load_page.submitted.connect(self._on_submit)
        self.workspace_page.export_btn.clicked.connect(self._on_export_clicked)
        self.workspace_page.reset_btn.clicked.connect(self._on_reset)

    def _build_menu_and_toolbar(self):
        menu_bar = self.menuBar()

        # Settings action is shared between the File menu and the toolbar,
        # so it's built once, up front.
        self.settings_action = QAction("&Settings…", self)
        self.settings_action.setShortcut(QKeySequence("Ctrl+,"))
        self.settings_action.triggered.connect(self._on_open_settings)

        # --- File menu ---
        file_menu = menu_bar.addMenu("&File")
        open_action = QAction("&Open Image…", self)
        open_action.setShortcut(QKeySequence.Open)  # Ctrl+O
        open_action.triggered.connect(self._on_open_image)
        file_menu.addAction(open_action)

        self.export_action = QAction("&Export Results…", self)
        self.export_action.setShortcut(QKeySequence.Save)  # Ctrl+S
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self._on_export_clicked)
        file_menu.addAction(self.export_action)

        file_menu.addSeparator()
        file_menu.addAction(self.settings_action)
        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- Edit menu ---
        edit_menu = menu_bar.addMenu("&Edit")

        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)  # Ctrl+Z
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self._on_undo)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self._on_redo)
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()

        self.reset_edits_action = QAction("Reset &Edits", self)
        self.reset_edits_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self.reset_edits_action.setEnabled(False)
        self.reset_edits_action.setToolTip(
            "Discard manual landmark adjustments and restore the AI's original result"
        )
        self.reset_edits_action.triggered.connect(self._on_reset_edits)
        edit_menu.addAction(self.reset_edits_action)

        edit_menu.addSeparator()

        self.reset_action = QAction("Full &Reset…", self)
        self.reset_action.setShortcut(QKeySequence("Ctrl+R"))
        self.reset_action.setToolTip("Clear the loaded image and analysis, and return to the start")
        self.reset_action.triggered.connect(self._on_reset)
        edit_menu.addAction(self.reset_action)

        # --- View menu ---
        view_menu = menu_bar.addMenu("&View")
        self.zoom_in_action = QAction("Zoom &In", self)
        self.zoom_in_action.setShortcut(QKeySequence.ZoomIn)  # Ctrl++
        self.zoom_in_action.triggered.connect(lambda: self.workspace_page.canvas.zoom_in())
        self.zoom_out_action = QAction("Zoom &Out", self)
        self.zoom_out_action.setShortcut(QKeySequence.ZoomOut)  # Ctrl+-
        self.zoom_out_action.triggered.connect(lambda: self.workspace_page.canvas.zoom_out())
        self.fit_action = QAction("&Fit to View", self)
        self.fit_action.setShortcut(QKeySequence("Ctrl+0"))
        self.fit_action.triggered.connect(lambda: self.workspace_page.canvas.fit_in_view())

        self.edit_mode_action = QAction("&Edit Mode", self)
        self.edit_mode_action.setShortcut(QKeySequence("Ctrl+E"))
        self.edit_mode_action.setCheckable(True)
        self.edit_mode_action.setEnabled(False)
        self.edit_mode_action.setToolTip("Available once landmarks are loaded")
        self.edit_mode_action.toggled.connect(self._on_edit_mode_toggled)

        self.retry_action = QAction("&Retry AI Analysis", self)
        self.retry_action.setShortcut(QKeySequence("F5"))
        self.retry_action.setEnabled(False)
        self.retry_action.triggered.connect(self._on_retry_analysis)

        for action in (self.zoom_in_action, self.zoom_out_action, self.fit_action,
                       self.edit_mode_action, self.retry_action):
            view_menu.addAction(action)

        # --- Tools menu (ML-team QA workflow, unrelated to the clinical state) ---
        tools_menu = menu_bar.addMenu("&Tools")
        self.validation_action = QAction("&Model Validation…", self)
        self.validation_action.triggered.connect(self._on_open_validation)
        tools_menu.addAction(self.validation_action)

        # --- Toolbar: zoom / edit / undo-redo / reset-edits / settings, far right ---
        self.toolbar = QToolBar("Main Toolbar", self)
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Without this, the spacer picks up the theme's generic QWidget
        # background rule (a shade darker than the toolbar itself) and
        # renders as a visible empty bar instead of blending in.
        spacer.setStyleSheet("background: transparent;")
        self.toolbar.addWidget(spacer)

        self.toolbar.addAction(self.zoom_out_action)
        self.toolbar.addAction(self.zoom_in_action)
        self.toolbar.addAction(self.fit_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.edit_mode_action)
        self.toolbar.addAction(self.undo_action)
        self.toolbar.addAction(self.redo_action)
        self.toolbar.addAction(self.reset_edits_action)
        self.toolbar.addSeparator()

        self.toolbar.addAction(self.settings_action)

        # Zoom/fit only make sense once an image is on the canvas.
        for action in (self.zoom_in_action, self.zoom_out_action, self.fit_action):
            action.setEnabled(False)

    # ------------------------------------------------------------------
    # Load / Submit
    # ------------------------------------------------------------------

    def _on_open_image(self):
        """File -> Open Image: shows the Load page and opens its file picker,
        so the same validation/preview/Submit flow always applies."""
        self.stack.setCurrentWidget(self.load_page)
        self.load_page.drop_zone._on_browse()

    def _on_submit(self, image_path):
        self.image_path = image_path
        pixmap = QPixmap(image_path)
        # Overlay items are owned by the graphics scene.  Clear our tracked
        # references before ImageCanvas.load_image() calls scene.clear(),
        # which deletes the underlying C++ QGraphicsItems.
        self.overlay_layer.clear()
        self.workspace_page.canvas.load_image(pixmap)
        self.workspace_page.reset_metrics()
        self.stack.setCurrentWidget(self.workspace_page)

        for action in (self.zoom_in_action, self.zoom_out_action, self.fit_action):
            action.setEnabled(True)

        self.statusBar().showMessage(f"Loaded: {os.path.basename(image_path)} — running AI analysis…")
        self._run_inference(image_path)

    # ------------------------------------------------------------------
    # Backend inference
    # ------------------------------------------------------------------

    def _run_inference(self, image_path):
        api_url = SettingsDialog.get_saved_api_url()
        self.retry_action.setEnabled(False)
        worker = InferenceWorker(image_path, api_url=api_url, timeout=INFERENCE_TIMEOUT, parent=self)
        worker.succeeded.connect(self._on_inference_succeeded)
        worker.failed.connect(self._on_inference_failed)
        self._inference_worker = worker
        worker.start()

    def _on_retry_analysis(self):
        if self.image_path:
            self.statusBar().showMessage("Retrying AI analysis…")
            self._run_inference(self.image_path)

    def _on_inference_succeeded(self, data, elapsed):
        if self.sender() is not self._inference_worker:
            return  # stale result from a superseded request (e.g. after Reset)

        pixmap = QPixmap(self.image_path)
        self.model_engine = ScoliosisModelEngine(autoload=False)
        self.model_engine.load_from_dict(data)
        self.model_engine.scale_coordinates(pixmap.width(), pixmap.height())
        # Capture the AI's original result *after* scaling, so "Reset Edits"
        # restores coordinates in the same space the canvas actually displays.
        self.model_engine.capture_baseline()

        self.overlay_layer.clear()
        self.overlay_layer.render(self.model_engine)

        self.edit_mode_action.setEnabled(True)
        self.edit_mode_action.setToolTip("")
        self.retry_action.setEnabled(False)
        self.workspace_page.export_btn.setEnabled(True)
        self.workspace_page.export_btn.setToolTip("")
        self.export_action.setEnabled(True)
        self._dirty = False
        self._update_edit_action_states()

        self._populate_metrics(elapsed)

        count = len(self.model_engine.get_detections())
        self.statusBar().showMessage(f"Analysis complete — {count} vertebrae detected.")

    def _on_inference_failed(self, message):
        if self.sender() is not self._inference_worker:
            return

        self.retry_action.setEnabled(True)
        self.statusBar().showMessage("AI analysis unavailable — showing image only.")
        QMessageBox.warning(
            self, "AI Analysis Unavailable",
            f"{message}\n\nYou can still view and zoom the image. "
            "Use View → Retry AI Analysis once the backend is reachable."
        )

    # ------------------------------------------------------------------
    # Edit mode / landmark dragging / undo-redo
    # ------------------------------------------------------------------

    def _on_edit_mode_toggled(self, enabled):
        self.overlay_layer.set_interactive(enabled)

    def _on_drag_started(self, det_idx, kp_idx):
        """Fired once per drag gesture (mouse press on a handle) -- this is
        the undo checkpoint, capturing the state right before this specific
        adjustment begins."""
        if self.model_engine is None:
            return
        self.model_engine.snapshot_for_undo()
        self._update_edit_action_states()

    def _on_keypoint_dragged(self, det_idx, kp_idx, x, y):
        """Fired continuously while a handle is being dragged (every mouse-
        move tick). Keeps the model data and the *cheap* parts of the
        overlay (outline polygon, Cobb lines/labels, CSVL) live-updated. See
        modules/overlay.py's render()/​_render_cobb_overlays() for why this
        no longer tears down and rebuilds the whole overlay on every tick --
        that reentrant scene-mutation pattern was the cause of the
        freeze/crash when dragging a landmark."""
        if self.model_engine is None:
            return
        self.model_engine.update_keypoint(det_idx, kp_idx, x, y)
        self.overlay_layer.render(self.model_engine)
        self._populate_metrics(elapsed=None)
        self._dirty = True

    def _on_drag_finished(self, det_idx, kp_idx):
        """Fired once per drag gesture (mouse release). The live updates
        during the drag already keep everything consistent, but this is a
        good, safe point (mouse grab has ended) to refresh button states."""
        self._update_edit_action_states()

    def _on_undo(self):
        if self.model_engine is None or not self.model_engine.undo():
            return
        self.overlay_layer.clear()
        self.overlay_layer.render(self.model_engine)
        self._populate_metrics(elapsed=None)
        self._dirty = self.model_engine.has_edits()
        self._update_edit_action_states()
        self.statusBar().showMessage("Undid last landmark adjustment.")

    def _on_redo(self):
        if self.model_engine is None or not self.model_engine.redo():
            return
        self.overlay_layer.clear()
        self.overlay_layer.render(self.model_engine)
        self._populate_metrics(elapsed=None)
        self._dirty = self.model_engine.has_edits()
        self._update_edit_action_states()
        self.statusBar().showMessage("Redid landmark adjustment.")

    def _on_reset_edits(self):
        if self.model_engine is None:
            return
        if not self.model_engine.has_edits():
            return
        if not self._confirm_discard_if_dirty("resetting your manual edits"):
            return
        self.model_engine.reset_edits()
        self.overlay_layer.clear()
        self.overlay_layer.render(self.model_engine)
        self._populate_metrics(elapsed=None)
        self._dirty = False
        self._update_edit_action_states()
        self.statusBar().showMessage("Manual edits reset to the AI's original result.")

    def _update_edit_action_states(self):
        engine = self.model_engine
        has_engine = engine is not None
        self.undo_action.setEnabled(has_engine and engine.can_undo())
        self.redo_action.setEnabled(has_engine and engine.can_redo())
        self.reset_edits_action.setEnabled(has_engine and engine.has_edits())

    # ------------------------------------------------------------------
    # Measurement panel
    # ------------------------------------------------------------------

    def _populate_metrics(self, elapsed):
        engine = self.model_engine
        if engine is None:
            return
        metrics = self.workspace_page.metrics

        metrics["Primary Cobb Angle"].set_value(f"{engine.get_selected_cobb_angle():.1f}°")

        pairs = engine.get_angle_pairs()
        metrics["Curve 1"].set_value(f"{pairs[0]['cobb_angle']:.1f}°" if len(pairs) > 0 else "—")
        metrics["Curve 2"].set_value(f"{pairs[1]['cobb_angle']:.1f}°" if len(pairs) > 1 else "—")

        apex_idx, deviation_px = engine.get_apex()
        if apex_idx is not None:
            metrics["Apex"].set_value(f"Vertebra #{apex_idx}")
            metrics["CSVL Deviation"].set_value(f"{deviation_px:.1f} px")
        else:
            metrics["Apex"].set_value("—")
            metrics["CSVL Deviation"].set_value("—")

        metrics["Vertebrae"].set_value(str(len(engine.get_detections())))

        if elapsed is not None:
            metrics["Processing Time"].set_value(f"{elapsed:.2f}s (API round-trip)")

    # ------------------------------------------------------------------
    # Reset / Settings / Export / Validation
    # ------------------------------------------------------------------

    def _on_reset(self):
        if not self._confirm_discard_if_dirty("resetting"):
            return

        self.image_path = None
        self.model_engine = None
        self._inference_worker = None
        self._dirty = False
        self.overlay_layer.clear()
        self.load_page.reset()
        self.workspace_page.reset_metrics()
        self.workspace_page.export_btn.setEnabled(False)
        self.workspace_page.export_btn.setToolTip("Available once AI analysis results are loaded")
        self.export_action.setEnabled(False)
        self.stack.setCurrentWidget(self.load_page)

        self.edit_mode_action.setChecked(False)
        self.edit_mode_action.setEnabled(False)
        self.retry_action.setEnabled(False)
        self._update_edit_action_states()

        for action in (self.zoom_in_action, self.zoom_out_action, self.fit_action):
            action.setEnabled(False)

        self.statusBar().showMessage("Load a spine X-ray image to begin.")

    def _on_open_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec()

    def _on_export_clicked(self):
        if self.model_engine is None:
            return
        dialog = ExportDialog(self.model_engine, self)
        if dialog.exec() == QDialog.Accepted:
            self._dirty = False

    def _on_open_validation(self):
        dialog = ValidationDialog(self)
        dialog.exec()

    def _confirm_discard_if_dirty(self, action_description):
        """Asks before throwing away landmark adjustments that haven't been
        exported yet. Returns True if it's OK to proceed."""
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Adjustments",
            f"You've adjusted landmarks since the last export.\n\n"
            f"Continue with {action_description} and discard these changes?",
            QMessageBox.Yes | QMessageBox.No
        )
        return reply == QMessageBox.Yes

    def closeEvent(self, event):
        """Guards against closing the window with unexported landmark
        adjustments still pending."""
        if self._confirm_discard_if_dirty("closing"):
            event.accept()
        else:
            event.ignore()
