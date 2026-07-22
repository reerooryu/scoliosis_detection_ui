# Main application window: single-page clinical workspace.
#
# Owns the toolbar (custom clinical theme) and the native menu bar
# (File/Edit/View/Tools), and swaps between the Load page (image import)
# and the Workspace page (canvas + measurement panel) via a QStackedWidget.
#
# The actual workflow logic -- submit/retry/reset, the background inference
# request lifecycle, landmark-edit operations (drag/undo/redo/reset edits),
# and project save/open (modules/project.py) -- lives in
# modules/controller.py:AnalysisController, driving an
# modules/session.py:AnalysisSession. This window's job is composition root,
# navigation, and dialog factory: it builds the menu/toolbar/pages, wires
# widget signals to controller methods, and binds session/controller signals
# to widget updates, rather than manually refreshing every widget itself
# after each operation. See AGENTS.md for the full rationale.
#
# Tools > Model Validation opens a separate, unrelated workflow
# (modules/validation.py) for comparing a model prediction against a
# ground-truth label file -- that's an ML-team QA task, not something a
# clinician does per-patient, so it's deliberately kept out of this
# window's own state.

import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFrame,
    QLabel, QPushButton, QStackedWidget, QToolBar, QSizePolicy, QStatusBar,
    QMessageBox, QDialog, QFileDialog
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

        self.export_btn = QPushButton("Export...")
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

        # Custom clinical theme is applied only to these specific content
        # widgets -- the menu bar and every dialog stay native OS style.
        theme.apply_clinical_theme(self.toolbar, self.stack, self.statusBar())

        self.overlay_layer = OverlayLayer(
            self.workspace_page.canvas,
            cobb_line_color=SettingsDialog.get_saved_line_color(),
        )

        # --- State (AnalysisSession) + workflow (AnalysisController) -------
        self.session = AnalysisSession(self)
        self.controller = AnalysisController(
            self.session,
            self.overlay_layer,
            api_url_provider=SettingsDialog.get_saved_api_url,
            timeout=INFERENCE_TIMEOUT,
            parent=self,
        )

        # Session state -> widget bindings. The window never manually
        # refreshes a widget mid-operation; it only reacts to these.
        self.session.state_changed.connect(self._on_state_changed)
        self.session.metrics_changed.connect(self._on_metrics_changed)
        self.session.edit_state_changed.connect(self._on_edit_state_changed)

        # Controller -> window bindings (status text, dialogs, the one
        # metric -- processing time -- that isn't part of persistent state).
        self.controller.status_message.connect(self.statusBar().showMessage)
        self.controller.error_dialog.connect(self._on_error_dialog)
        self.controller.analysis_completed.connect(self._on_analysis_completed)

        # Overlay drag signals -> controller (landmark edit workflow).
        self.overlay_layer.signals.keypoint_moved.connect(self.controller.on_keypoint_dragged)
        self.overlay_layer.signals.drag_started.connect(self.controller.on_drag_started)
        self.overlay_layer.signals.drag_finished.connect(self.controller.on_drag_finished)

        # Widget actions -> controller.
        self.load_page.submitted.connect(self._on_submit)
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

        # Save Project persists progress on an in-progress assessment (image
        # + current + baseline detections) so it can be resumed later without
        # re-running AI inference -- distinct from Export Results below,
        # which is a one-way clinical deliverable, not something you reopen
        # and keep editing. See modules/project.py.
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
        if not self._confirm_discard_if_dirty("loading a new image"):
            return
        self.stack.setCurrentWidget(self.load_page)
        self.load_page.drop_zone._on_browse()

    def _on_submit(self, image_path):
        """The controller clears the overlay and cancels any outstanding
        request as part of submit() -- that must happen before
        canvas.load_image() below, since ImageCanvas.load_image() calls
        scene.clear(), which deletes the overlay's C++ items out from under
        any still-tracked Python references."""
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

    def _on_metrics_changed(self):
        engine = self.session.model_engine
        if engine is None:
            return
        metrics = self.workspace_page.metrics

        metrics["Primary Cobb Angle"].set_value("{:.1f} deg".format(engine.get_selected_cobb_angle()))

        pairs = engine.get_angle_pairs()
        metrics["Curve 1"].set_value("{:.1f} deg".format(pairs[0]["cobb_angle"]) if len(pairs) > 0 else "-")
        metrics["Curve 2"].set_value("{:.1f} deg".format(pairs[1]["cobb_angle"]) if len(pairs) > 1 else "-")

        apex_idx, deviation_px = engine.get_apex()
        if apex_idx is not None:
            metrics["Apex"].set_value("Vertebra #{}".format(apex_idx))
            metrics["CSVL Deviation"].set_value("{:.1f} px".format(deviation_px))
        else:
            metrics["Apex"].set_value("-")
            metrics["CSVL Deviation"].set_value("-")

        metrics["Vertebrae"].set_value(str(len(engine.get_detections())))

    def _on_analysis_completed(self, elapsed):
        self.workspace_page.metrics["Processing Time"].set_value("{:.2f}s (API round-trip)".format(elapsed))

    def _on_edit_state_changed(self):
        self.undo_action.setEnabled(self.session.can_undo())
        self.redo_action.setEnabled(self.session.can_redo())
        self.reset_edits_action.setEnabled(self.session.has_edits())

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
        dialog = ExportDialog(self.session.model_engine, self)
        if dialog.exec() == QDialog.Accepted:
            self.session.dirty = False

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
        if self.controller.open_project(path):
            self.stack.setCurrentWidget(self.workspace_page)
            # If the workspace page has never been shown before in this run
            # of the app, its canvas won't have a settled viewport size the
            # instant setCurrentWidget() returns -- and ImageCanvas itself
            # already accounts for this for the base image (see its own
            # showEvent/_delayed_initial_fit, on a 100ms timer). The
            # overlay's Cobb-label placement depends on that same settled
            # viewport, so render it just after ImageCanvas's own delayed
            # re-fit would have fired, rather than synchronously right now
            # (see AnalysisController.open_project()'s docstring for why
            # the Submit flow never needs this: its network round-trip
            # already provides the delay for free).
            QTimer.singleShot(150, self.controller.finish_open_project)

    def _on_save_project(self):
        if self.session.project_path:
            self.controller.save_project(self.session.project_path)
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

    def _on_open_validation(self):
        dialog = ValidationDialog(self)
        dialog.exec()

    def _confirm_discard_if_dirty(self, action_description):
        """Asks before throwing away landmark adjustments that haven't been
        exported or saved to a project yet. Returns True if it's OK to
        proceed. session.dirty and session.project_dirty are independent
        (you can export without saving a project, or save a project without
        exporting), so either one being set means there's something to lose."""
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
        """Guards against closing the window with unexported and/or
        unsaved-to-project landmark adjustments still pending."""
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
