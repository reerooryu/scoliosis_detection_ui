# Main window: the native menu bar, the themed toolbar, and a stack that
# switches between the Load page and the Workspace page.
#
# The workflow itself (inference, edits, project save/open) lives in
# AnalysisController (modules/controller.py), which updates an
# AnalysisSession (modules/session.py). This window builds the widgets,
# passes user actions to the controller, and refreshes itself from the
# session's and controller's signals.

import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFrame,
    QLabel, QPushButton, QStackedWidget, QToolBar, QSizePolicy, QStatusBar,
    QMessageBox, QDialog, QFileDialog, QProgressBar
)

from config import APP_NAME, WINDOW_WIDTH, WINDOW_HEIGHT, INFERENCE_TIMEOUT
from modules import theme
from modules.load_view import LoadPage
from modules.canvas import ImageCanvas
from modules.overlay import OverlayLayer
from modules.session import AnalysisSession
from modules.controller import AnalysisController
from modules.settings_dialog import SettingsDialog
from modules.export import ExportDialog
from modules.validation import ValidationDialog
from modules.project import PROJECT_EXTENSION


class MetricRow(QFrame):
    """A single labeled measurement value in the right-hand panel."""

    def __init__(self, label, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(2)

        lbl = QLabel(label)
        lbl.setObjectName("MetricLabel")
        self.value_lbl = QLabel("-")
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
        # Needed for the stylesheet background to paint (see LoadPage).
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

        self.export_btn = QPushButton("Export...")
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip("Available once AI analysis results are loaded")
        panel_layout.addWidget(self.export_btn)

        # Full Reset lives here; the toolbar has "Reset Edits" instead.
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setToolTip("Clear the loaded image and analysis, and return to the start")
        panel_layout.addWidget(self.reset_btn)

        splitter.addWidget(panel)
        splitter.setSizes([1000, 300])

    def reset_metrics(self):
        for row in self.metrics.values():
            row.set_value("-")


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self._build_menu_and_toolbar()

        self.stack = QStackedWidget(self)
        self.load_page = LoadPage(self)
        self.workspace_page = WorkspacePage(self)
        self.stack.addWidget(self.load_page)
        self.stack.addWidget(self.workspace_page)
        self.setCentralWidget(self.stack)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Load a spine X-ray image to begin.")

        # Moving bar in the status bar while an AI request is running.
        self.busy_bar = QProgressBar(self)
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setFixedSize(140, 12)
        self.busy_bar.hide()
        self.statusBar().addPermanentWidget(self.busy_bar)

        # Custom clinical theme is applied only to these specific content
        # widgets -- the menu bar and every dialog stay native OS style.
        theme.apply_clinical_theme(self.toolbar, self.stack, self.statusBar())

        self.overlay_layer = OverlayLayer(
            self.workspace_page.canvas,
            cobb_line_color=SettingsDialog.get_saved_line_color(),
        )
        # Cobb labels are placed in screen pixels, so re-place them on zoom.
        self.workspace_page.canvas.view_changed.connect(self.overlay_layer.reposition_labels)

        # --- State (AnalysisSession) + workflow (AnalysisController) -------
        self.session = AnalysisSession(self)
        self.controller = AnalysisController(
            self.session,
            self.overlay_layer,
            api_url_provider=SettingsDialog.get_saved_api_url,
            timeout=INFERENCE_TIMEOUT,
            parent=self,
        )

        # Session signals -> widget updates.
        self.session.state_changed.connect(self._on_state_changed)
        self.session.metrics_changed.connect(self._on_metrics_changed)
        self.session.edit_state_changed.connect(self._on_edit_state_changed)

        # Controller signals -> status text, error dialogs, processing time.
        self.controller.status_message.connect(self.statusBar().showMessage)
        self.controller.error_dialog.connect(self._on_error_dialog)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.busy_changed.connect(self.busy_bar.setVisible)

        # Overlay drag signals -> controller (landmark edit workflow).
        self.overlay_layer.signals.keypoint_moved.connect(self.controller.on_keypoint_dragged)
        self.overlay_layer.signals.drag_started.connect(self.controller.on_drag_started)
        self.overlay_layer.signals.drag_finished.connect(self.controller.on_drag_finished)

        # Widget actions -> controller.
        self.load_page.submitted.connect(self._on_submit)
        self.load_page.project_opened.connect(self._on_load_page_project_opened)
        self.workspace_page.export_btn.clicked.connect(self._on_export_clicked)
        self.workspace_page.reset_btn.clicked.connect(self._on_reset)

    def _build_menu_and_toolbar(self):
        menu_bar = self.menuBar()

        # Settings action is shared between the File menu and the toolbar,
        # so it's built once, up front.
        self.settings_action = QAction("&Settings...", self)
        self.settings_action.setShortcut(QKeySequence("Ctrl+,"))
        self.settings_action.triggered.connect(self._on_open_settings)

        # --- File menu ---
        file_menu = menu_bar.addMenu("&File")
        open_action = QAction("&Open Image...", self)
        open_action.setShortcut(QKeySequence.Open)  # Ctrl+O
        open_action.triggered.connect(self._on_open_image)
        file_menu.addAction(open_action)

        self.open_project_action = QAction("Open &Project...", self)
        self.open_project_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.open_project_action.triggered.connect(self._on_open_project)
        file_menu.addAction(self.open_project_action)

        file_menu.addSeparator()

        # A saved project can be reopened and edited later; Export (below)
        # is a one-way output.
        self.save_project_action = QAction("&Save Project", self)
        self.save_project_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.save_project_action.setEnabled(False)
        self.save_project_action.triggered.connect(self._on_save_project)
        file_menu.addAction(self.save_project_action)

        self.save_project_as_action = QAction("Save Project &As...", self)
        self.save_project_as_action.setEnabled(False)
        self.save_project_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(self.save_project_as_action)

        file_menu.addSeparator()

        self.export_action = QAction("&Export Results...", self)
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

        self.reset_action = QAction("Full &Reset...", self)
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
        self.validation_action = QAction("&Model Validation...", self)
        self.validation_action.triggered.connect(self._on_open_validation)
        tools_menu.addAction(self.validation_action)

        # --- Toolbar: zoom / edit / undo-redo / reset-edits / settings, far right ---
        self.toolbar = QToolBar("Main Toolbar", self)
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Transparent, or the spacer shows as a darker bar in the toolbar.
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
        """File -> Open Image: show the Load page and open its file picker."""
        if not self._confirm_discard_if_dirty("loading a new image"):
            return
        previous_page = self.stack.currentWidget()
        self.stack.setCurrentWidget(self.load_page)
        staged = self.load_page.drop_zone._on_browse()
        if not staged and self.stack.currentWidget() is self.load_page:
            # Nothing was picked: go back, so a loaded assessment stays visible.
            self.stack.setCurrentWidget(previous_page)

    def _on_submit(self, image_path):
        """controller.submit() must run before canvas.load_image(): it clears
        the overlay, and load_image() then wipes the scene."""
        self.controller.submit(image_path)

        pixmap = QPixmap(image_path)
        self.workspace_page.canvas.load_image(pixmap)
        self.stack.setCurrentWidget(self.workspace_page)

    def _on_retry_analysis(self):
        self.controller.retry()

    # ------------------------------------------------------------------
    # Session / controller signal handlers
    # ------------------------------------------------------------------

    def _on_state_changed(self, state):
        has_image = state != AnalysisSession.STATE_EMPTY
        is_ready = state == AnalysisSession.STATE_READY
        is_error = state == AnalysisSession.STATE_ERROR

        for action in (self.zoom_in_action, self.zoom_out_action, self.fit_action):
            action.setEnabled(has_image)

        self.edit_mode_action.setEnabled(is_ready)
        self.edit_mode_action.setToolTip("" if is_ready else "Available once landmarks are loaded")

        self.workspace_page.export_btn.setEnabled(is_ready)
        self.workspace_page.export_btn.setToolTip(
            "" if is_ready else "Available once AI analysis results are loaded"
        )
        self.export_action.setEnabled(is_ready)
        self.save_project_action.setEnabled(is_ready)
        self.save_project_as_action.setEnabled(is_ready)

        self.retry_action.setEnabled(is_error)

        if state in (AnalysisSession.STATE_EMPTY, AnalysisSession.STATE_LOADING):
            self.workspace_page.reset_metrics()

        if state == AnalysisSession.STATE_EMPTY:
            self.edit_mode_action.setChecked(False)
            self.stack.setCurrentWidget(self.load_page)

        if is_error:
            self.statusBar().showMessage("AI analysis unavailable -- showing image only.")
        elif state == AnalysisSession.STATE_EMPTY:
            self.statusBar().showMessage("Load a spine X-ray image to begin.")

        self._update_window_title()

    def _on_metrics_changed(self):
        engine = self.session.model_engine
        if engine is None:
            return
        metrics = self.workspace_page.metrics

        metrics["Primary Cobb Angle"].set_value("{:.1f}°".format(engine.get_selected_cobb_angle()))

        pairs = engine.get_angle_pairs()
        metrics["Curve 1"].set_value("{:.1f}°".format(pairs[0]["cobb_angle"]) if len(pairs) > 0 else "-")
        metrics["Curve 2"].set_value("{:.1f}°".format(pairs[1]["cobb_angle"]) if len(pairs) > 1 else "-")

        apex_idx, deviation_px = engine.get_apex()
        if apex_idx is not None:
            metrics["Apex"].set_value("Vertebra #{}".format(apex_idx))
            metrics["CSVL Deviation"].set_value("{:.1f} px".format(deviation_px))
        else:
            metrics["Apex"].set_value("-")
            metrics["CSVL Deviation"].set_value("-")

        metrics["Vertebrae"].set_value(str(len(engine.get_detections())))
        self._update_window_title()

    def _on_analysis_completed(self, elapsed):
        self.workspace_page.metrics["Processing Time"].set_value("{:.2f}s (API round-trip)".format(elapsed))

    def _on_edit_state_changed(self):
        self.undo_action.setEnabled(self.session.can_undo())
        self.redo_action.setEnabled(self.session.can_redo())
        self.reset_edits_action.setEnabled(self.session.has_edits())
        self._update_window_title()

    def _source_name(self):
        """File name of the X-ray being assessed, or None."""
        if self.session.original_filename:
            return self.session.original_filename
        if self.session.image_path:
            return os.path.basename(self.session.image_path)
        return None

    def _update_window_title(self):
        """Show the open project or image in the title. It is marked as
        modified while a project has unsaved edits, or an image without a
        project has unexported edits."""
        if self.session.project_path:
            name = os.path.basename(self.session.project_path)
            modified = self.session.project_dirty
        else:
            name = self._source_name()
            modified = self.session.dirty
        if self.session.state == AnalysisSession.STATE_EMPTY or not name:
            self.setWindowModified(False)
            self.setWindowTitle(APP_NAME)
            return
        self.setWindowTitle("{}[*] - {}".format(name, APP_NAME))
        self.setWindowModified(modified)

    def _on_error_dialog(self, title, message):
        QMessageBox.warning(self, title, message)

    # ------------------------------------------------------------------
    # Edit mode / landmark editing
    # ------------------------------------------------------------------

    def _on_edit_mode_toggled(self, enabled):
        self.overlay_layer.set_interactive(enabled)

    def _on_undo(self):
        self.controller.undo()

    def _on_redo(self):
        self.controller.redo()

    def _on_reset_edits(self):
        if not self.session.has_edits():
            return
        if not self._confirm_discard_if_dirty("resetting your manual edits"):
            return
        self.controller.reset_edits()

    # ------------------------------------------------------------------
    # Reset / Settings / Export / Validation
    # ------------------------------------------------------------------

    def _on_reset(self):
        if not self._confirm_discard_if_dirty("resetting"):
            return
        self.controller.reset()
        self.load_page.reset()

    def _on_open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self.overlay_layer.set_cobb_line_color(SettingsDialog.get_saved_line_color())

    def _on_export_clicked(self):
        if self.session.model_engine is None:
            return
        dialog = ExportDialog(self.session.model_engine, self, source_name=self._source_name())
        if dialog.exec() == QDialog.Accepted:
            self.session.dirty = False
            self._update_window_title()

    # ------------------------------------------------------------------
    # Project save / open (modules/project.py, modules/controller.py)
    # ------------------------------------------------------------------

    def _on_open_project(self):
        if not self._confirm_discard_if_dirty("opening a different project"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", f"Scoliosis Project Files (*{PROJECT_EXTENSION})"
        )
        if not path:
            return
        self._open_project_path(path)

    def _on_load_page_project_opened(self, path):
        """A .sdproj dropped or browsed to on the Load page opens the same
        way as File -> Open Project."""
        if not self._confirm_discard_if_dirty("opening a different project"):
            return
        self._open_project_path(path)

    def _open_project_path(self, path):
        if self.controller.open_project(path):
            self.stack.setCurrentWidget(self.workspace_page)
            # No inference ran, so clear any time left from an earlier analysis.
            self.workspace_page.metrics["Processing Time"].set_value("-")
            # Draw the overlay once the page is visible and laid out; label
            # placement needs the final viewport size.
            QTimer.singleShot(150, self.controller.finish_open_project)

    def _on_save_project(self):
        if self.session.project_path:
            self.controller.save_project(self.session.project_path)
            self._update_window_title()
        else:
            self._on_save_project_as()

    def _on_save_project_as(self):
        default_name = "assessment"
        if self.session.original_filename:
            default_name = os.path.splitext(self.session.original_filename)[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", default_name + PROJECT_EXTENSION,
            f"Scoliosis Project Files (*{PROJECT_EXTENSION})"
        )
        if not path:
            return
        if not path.endswith(PROJECT_EXTENSION):
            path += PROJECT_EXTENSION
        self.controller.save_project(path)
        self._update_window_title()

    def _on_open_validation(self):
        dialog = ValidationDialog(self)
        dialog.exec()

    def _confirm_discard_if_dirty(self, action_description):
        """Ask before discarding edits that are neither exported nor saved
        to a project. Returns True if it is OK to proceed."""
        if not (self.session.dirty or self.session.project_dirty):
            return True
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "You've made changes that haven't been exported or saved to a project.\n\n"
            "Continue with {} and discard these changes?".format(action_description),
            QMessageBox.Yes | QMessageBox.No
        )
        return reply == QMessageBox.Yes

    def closeEvent(self, event):
        """Block closing while a request is running, and confirm before
        discarding unsaved edits."""
        if self.controller.has_pending_jobs():
            QMessageBox.information(
                self,
                "Analysis Still Running",
                "Please wait for the active AI analysis request to finish before closing. "
                "The application will remain responsive while it completes.",
            )
            event.ignore()
            return
        if self._confirm_discard_if_dirty("closing"):
            event.accept()
        else:
            event.ignore()
