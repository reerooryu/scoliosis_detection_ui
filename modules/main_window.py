# Main window: the native menu bar, the themed toolbar, and a stack that
# switches between the Load page and the Workspace page.
#
# The workflow itself (inference, edits, project save/open) lives in
# AnalysisController (modules/controller.py), which updates an
# AnalysisSession (modules/session.py). This window builds the widgets,
# passes user actions to the controller, and refreshes itself from the
# session's and controller's signals.

import logging
import os
import threading

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QPixmap, QAction, QKeySequence, QDesktopServices
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFrame,
    QLabel, QPushButton, QStackedWidget, QToolBar, QSizePolicy, QStatusBar,
    QMessageBox, QDialog, QFileDialog, QProgressBar
)

from config import APP_NAME, APP_VERSION, WINDOW_WIDTH, WINDOW_HEIGHT, INFERENCE_TIMEOUT
from modules import theme
from modules.load_view import LoadPage
from modules.canvas import ImageCanvas
from modules.overlay import OverlayLayer
from modules.session import AnalysisSession
from modules.controller import AnalysisController
from modules import analysis, settings, updater, updates
from modules.settings import MODE_LOCAL
from modules.settings_dialog import SettingsDialog, choose_model_file
from modules.app_paths import LOG_PATH
from modules.export import ExportDialog
from modules.validation import ValidationDialog
from modules.project import PROJECT_EXTENSION
from modules.image_io import ImageLoadError, load_image

logger = logging.getLogger(__name__)


class MetricRow(QFrame):
    """A single labeled measurement value in the right-hand panel."""

    def __init__(self, label, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(2)

        # Small color square in front of the label; hidden unless set_color()
        # is called (used by the curve rows to match the lines on the image).
        self.color_chip = QLabel()
        self.color_chip.setFixedSize(10, 10)
        self.color_chip.hide()
        lbl = QLabel(label)
        lbl.setObjectName("MetricLabel")
        self.value_lbl = QLabel("-")
        self.value_lbl.setObjectName("MetricValue")

        header = QHBoxLayout()
        header.setSpacing(6)
        header.addWidget(self.color_chip)
        header.addWidget(lbl)
        header.addStretch()
        layout.addLayout(header)
        layout.addWidget(self.value_lbl)

    def set_value(self, text):
        self.value_lbl.setText(text)

    def set_color(self, color):
        self.color_chip.setStyleSheet("background-color: {}; border-radius: 2px;".format(color))
        self.color_chip.show()

    def set_tooltip(self, text):
        self.value_lbl.setToolTip(text)


class WorkspacePage(QWidget):
    """Canvas (left) + measurement summary and export action (right)."""

    # Curve 1 and 2 are always shown; Curve 3 and 4 only when they exist.
    MAX_CURVE_ROWS = 4
    METRIC_KEYS = [
        "Primary Cobb Angle", "Curve 1", "Curve 2", "Curve 3", "Curve 4", "Apex",
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
            "The vertebra furthest sideways from the CSVL. Levels are counted "
            "up from the lowest vertebra found, as L5."
        )
        self.metrics["CSVL Deviation"].set_tooltip(
            "The CSVL (Central Sacral Vertical Line) is drawn through the "
            "center of the lowest vertebra found, because the AI does not "
            "mark the sacrum. The distance is in pixels: the image has no "
            "pixel size, so it can't be converted to mm."
        )
        self.metrics["Processing Time"].set_tooltip(
            "Local: the AI run on this computer, without the one-time model "
            "load. Server: the full round trip to the server."
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
        self.show_curves([])

    def show_curves(self, texts):
        """Fill one row per curve; texts are the row values, top curve first."""
        for i in range(self.MAX_CURVE_ROWS):
            row = self.metrics["Curve {}".format(i + 1)]
            row.set_value(texts[i] if i < len(texts) else "-")
            row.setVisible(i < max(len(texts), 2))

    def set_curve_colors(self, colors):
        """Show each curve's color next to its row, matching the image."""
        for i in range(self.MAX_CURVE_ROWS):
            self.metrics["Curve {}".format(i + 1)].set_color(colors[i % len(colors)])

    def reset_metrics(self):
        for row in self.metrics.values():
            row.set_value("-")
        self.show_curves([])


class MainWindow(QMainWindow):
    # Sent from the update-check thread; Qt delivers it on the GUI thread.
    update_available = Signal(object)  # updates.UpdateInfo: offer the download page
    update_ready = Signal(object)      # updates.UpdateInfo: downloaded, ready to install
    update_progress = Signal(str, int) # (version, percent downloaded)
    update_paused = Signal(str)        # version: download stopped, continues next start

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

        # Far right of the status bar: update download progress (only while
        # there is something to say) and the app version.
        self.update_label = QLabel()
        self.update_label.setObjectName("MetricLabel")
        self.update_label.setContentsMargins(0, 0, 14, 0)
        self.update_label.hide()
        self.statusBar().addPermanentWidget(self.update_label)
        version_label = QLabel(f"v{APP_VERSION}")
        version_label.setObjectName("MetricLabel")
        version_label.setToolTip("App version. Help → About has more details.")
        version_label.setContentsMargins(0, 0, 6, 0)
        self.statusBar().addPermanentWidget(version_label)

        # Custom clinical theme is applied only to these specific content
        # widgets -- the menu bar and every dialog stay native OS style.
        theme.apply_clinical_theme(self.toolbar, self.stack, self.statusBar())

        curve_colors = settings.curve_colors()
        self.overlay_layer = OverlayLayer(self.workspace_page.canvas, curve_colors=curve_colors)
        self.workspace_page.set_curve_colors(curve_colors)
        # Cobb labels are placed in screen pixels, so re-place them on zoom.
        self.workspace_page.canvas.view_changed.connect(self.overlay_layer.reposition_labels)

        # --- State (AnalysisSession) + workflow (AnalysisController) -------
        self.session = AnalysisSession(self)
        self.controller = AnalysisController(
            self.session,
            self.overlay_layer,
            settings_provider=settings.inference_settings,
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
        self.controller.busy_changed.connect(self._on_busy_changed)

        # "Update Now" closes the window; app.py then installs the update.
        self.restart_for_update = False
        self._waiting_update = None   # a ready update waiting for the analysis to finish

        # Overlay drag signals -> controller (landmark edit workflow).
        self.overlay_layer.signals.keypoint_moved.connect(self.controller.on_keypoint_dragged)
        self.overlay_layer.signals.drag_started.connect(self.controller.on_drag_started)
        self.overlay_layer.signals.drag_finished.connect(self.controller.on_drag_finished)

        # Widget actions -> controller.
        self.load_page.submitted.connect(self._on_submit)
        self.load_page.project_opened.connect(self._on_load_page_project_opened)
        self.workspace_page.export_btn.clicked.connect(self._on_export_clicked)
        self.workspace_page.reset_btn.clicked.connect(self._on_reset)

        # Once the window is on screen: find the model file, start loading it.
        QTimer.singleShot(0, self._on_startup)

    # ------------------------------------------------------------------
    # Startup: the local model file
    # ------------------------------------------------------------------

    def _on_startup(self):
        self._start_update_check()
        self._ensure_model_file()
        self._preload_local_model()

    def _start_update_check(self):
        """Look for a newer release in the background, without interrupting
        the user. The packaged app also downloads and unpacks it there, then
        asks "Update Now / Later" (modules/updater.py). Run from source, or
        if the download fails, it offers the download page instead. No
        internet just means no message."""
        self.update_available.connect(self._on_update_available)
        self.update_ready.connect(self._on_update_ready)
        self.update_progress.connect(self._on_update_progress)
        self.update_paused.connect(self._on_update_paused)
        install = updater.current_install()

        def check():
            info = updates.check_for_update()
            if info is None:
                return
            if install is not None and info.asset is not None:
                last_percent = [-1]

                def progress(done, total):
                    percent = int(done * 100 / total) if total else 0
                    if percent != last_percent[0]:     # at most 100 signals
                        last_percent[0] = percent
                        self.update_progress.emit(info.version, percent)

                try:
                    updater.prepare_update(info, install, progress=progress)
                except updater.DownloadPaused as exc:
                    logger.info("Update %s: %s", info.version, exc)
                    self.update_paused.emit(info.version)
                    return
                except Exception as exc:
                    logger.warning("Preparing update %s failed: %s", info.version, exc)
                else:
                    self.update_ready.emit(info)
                    return
            self.update_available.emit(info)

        threading.Thread(target=check, name="update-check", daemon=True).start()

    def _on_update_ready(self, info):
        """The new version is downloaded. Ask now or later, but not while an
        analysis is running: then ask when it finishes."""
        self._show_update_status(f"Update v{info.version} ready")
        if self.controller.is_busy():
            self._waiting_update = info
            return
        if self._ask_update(
            "Update Ready", f"Scoliosis AI v{info.version} is ready to install.",
            f"You have v{APP_VERSION}.\n\n"
            "Update Now: the app closes and opens again in a few seconds.\n"
            "Later: the update installs by itself the next time you open the app.",
            info.notes, "Update Now",
        ):
            self.restart_for_update = True
            self.close()                 # asks about unsaved work first
            if self.isVisible():         # the user chose to keep working
                self.restart_for_update = False
        if not self.restart_for_update:
            self._show_update_status(f"Update v{info.version} installs next time you open the app")

    def _on_update_progress(self, version, percent):
        self._show_update_status(f"Downloading update v{version}… {percent}%")

    def _on_update_paused(self, version):
        self._show_update_status(f"Update v{version} download paused; it continues next time")

    def _show_update_status(self, text):
        """Show text in the status bar's update area ("" hides it)."""
        self.update_label.setText(text)
        self.update_label.setVisible(bool(text))

    def _on_busy_changed(self, busy):
        if not busy and self._waiting_update is not None:
            info, self._waiting_update = self._waiting_update, None
            # Let the result appear first.
            QTimer.singleShot(1500, lambda: self._on_update_ready(info))

    def _on_about(self):
        current = settings.inference_settings()
        where = ("this computer — " + (current.weights_path or "no model file chosen")
                 if current.mode == MODE_LOCAL else "server — " + current.api_url)
        QMessageBox.about(
            self, "About",
            f"{APP_NAME}\nVersion {APP_VERSION}\n\nAI model runs on: {where}\n\n"
            f"Log file (send it with any problem report; Help → Open Log Folder):\n{LOG_PATH}\n\n"
            "For research and testing only. Not for clinical decisions.",
        )

    def _on_open_log_folder(self):
        """Open the folder with the log file in Finder / File Explorer, so it
        can be sent with a problem report."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(LOG_PATH.parent)))

    def _on_update_available(self, info):
        """Fallback when the app can't update itself: offer the download page."""
        self._show_update_status("")
        if self._ask_update(
            "Update Available",
            f"A new version of the app is available: {info.version} (you have {APP_VERSION}).",
            "Open the download page? Download the zip for your computer and "
            "replace the old app folder with the new one. Your settings and "
            "model file are kept.",
            info.notes, "Open Download Page",
        ):
            QDesktopServices.openUrl(QUrl(info.url))

    def _ask_update(self, title, text, details, notes, accept_label):
        """An update question with accept_label / "Later". The release notes
        are under "Show Details". Returns True if accepted."""
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Information)
        box.setText(text)
        box.setInformativeText(details)
        if notes:
            box.setDetailedText(notes)
        accept_btn = box.addButton(accept_label, QMessageBox.AcceptRole)
        box.addButton("Later", QMessageBox.RejectRole)
        box.exec()
        return box.clickedButton() is accept_btn

    def _ensure_model_file(self):
        """Local mode needs a model file. If none is saved or found next to
        the app, ask for it once now rather than failing on the first image."""
        current = settings.inference_settings()
        if current.mode != MODE_LOCAL or current.weights_path:
            return
        QMessageBox.information(
            self, "Choose the AI Model File",
            "This app runs the AI on your computer and needs the model file "
            "(ScoliosisAI_model.pth) that came with it.\n\n"
            "Please choose that file in the next window. You can change it later "
            "in Settings → AI model.",
        )
        path = choose_model_file(self)
        if path:
            settings.save_weights_path(path)
        else:
            self.statusBar().showMessage(
                "No AI model file chosen — choose it in Settings before analysing an image."
            )

    def _preload_local_model(self):
        """Load the local model in the background, so the first analysis
        does not wait for it. A failure is only logged here; it is shown if
        the user then analyses an image."""
        current = settings.inference_settings()
        weights_path = current.weights_path
        if current.mode != MODE_LOCAL or not weights_path:
            return
        if analysis.is_local_model_loaded(weights_path):
            return

        def load():
            try:
                analysis.load_local_model(weights_path)
            except Exception as exc:
                logger.warning("Preloading the AI model failed: %s", exc)

        threading.Thread(target=load, name="preload-model", daemon=True).start()

    def _action(self, text, slot, shortcut=None, enabled=True, checkable=False, tooltip=None):
        """A menu/toolbar action. shortcut is a QKeySequence standard key or
        text such as "Ctrl+E"."""
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        action.setEnabled(enabled)
        action.setCheckable(checkable)
        if tooltip:
            action.setToolTip(tooltip)
        (action.toggled if checkable else action.triggered).connect(slot)
        return action

    def _build_menu_and_toolbar(self):
        # Settings is in the File menu and the toolbar.
        self.settings_action = self._action("&Settings...", self._on_open_settings, "Ctrl+,")

        # A saved project can be reopened and edited later; Export is a one-way output.
        self.open_project_action = self._action("Open &Project...", self._on_open_project, "Ctrl+Shift+O")
        self.save_project_action = self._action(
            "&Save Project", self._on_save_project, "Ctrl+Shift+S", enabled=False)
        self.save_project_as_action = self._action(
            "Save Project &As...", self._on_save_project_as, enabled=False)
        self.export_action = self._action(
            "&Export Results...", self._on_export_clicked, QKeySequence.Save, enabled=False)
        self._add_menu("&File", [
            self._action("&Open Image...", self._on_open_image, QKeySequence.Open),
            self.open_project_action, None,
            self.save_project_action, self.save_project_as_action, None,
            self.export_action, None,
            self.settings_action, None,
            self._action("E&xit", self.close, "Ctrl+Q"),
        ])

        self.undo_action = self._action("&Undo", self._on_undo, QKeySequence.Undo, enabled=False)
        self.redo_action = self._action("&Redo", self._on_redo, "Ctrl+Shift+Z", enabled=False)
        self.reset_edits_action = self._action(
            "Reset &Edits", self._on_reset_edits, "Ctrl+Shift+R", enabled=False,
            tooltip="Discard manual landmark adjustments and restore the AI's original result")
        self.reset_action = self._action(
            "Full &Reset...", self._on_reset, "Ctrl+R",
            tooltip="Clear the loaded image and analysis, and return to the start")
        self._add_menu("&Edit", [
            self.undo_action, self.redo_action, None, self.reset_edits_action, None, self.reset_action,
        ])

        # Zoom/fit only make sense once an image is on the canvas.
        self.zoom_in_action = self._action(
            "Zoom &In", lambda: self.workspace_page.canvas.zoom_in(), QKeySequence.ZoomIn, enabled=False)
        self.zoom_out_action = self._action(
            "Zoom &Out", lambda: self.workspace_page.canvas.zoom_out(), QKeySequence.ZoomOut, enabled=False)
        self.fit_action = self._action(
            "&Fit to View", lambda: self.workspace_page.canvas.fit_in_view(), "Ctrl+0", enabled=False)
        self.edit_mode_action = self._action(
            "&Edit Mode", self._on_edit_mode_toggled, "Ctrl+E", enabled=False, checkable=True,
            tooltip="Available once landmarks are loaded")
        self.retry_action = self._action(
            "&Retry AI Analysis", self._on_retry_analysis, "F5", enabled=False)
        self._add_menu("&View", [
            self.zoom_in_action, self.zoom_out_action, self.fit_action,
            self.edit_mode_action, self.retry_action,
        ])

        # ML-team QA workflow, unrelated to the clinical state.
        self._add_menu("&Tools", [self._action("&Model Validation...", self._on_open_validation)])

        about_action = self._action("&About", self._on_about)
        about_action.setMenuRole(QAction.AboutRole)  # goes in the app menu on a Mac
        self._add_menu("&Help", [
            about_action,
            self._action("Open &Log Folder", self._on_open_log_folder),
        ])

        # --- Toolbar: zoom / edit / undo-redo / reset-edits / settings, far right ---
        self.toolbar = QToolBar("Main Toolbar", self)
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Transparent, or the spacer shows as a darker bar in the toolbar.
        spacer.setStyleSheet("background: transparent;")
        self.toolbar.addWidget(spacer)

        for action in (self.zoom_out_action, self.zoom_in_action, self.fit_action, None,
                       self.edit_mode_action, self.undo_action, self.redo_action,
                       self.reset_edits_action, None, self.settings_action):
            if action is None:
                self.toolbar.addSeparator()
            else:
                self.toolbar.addAction(action)

    def _add_menu(self, title, actions):
        """A menu bar menu with these actions; None is a separator."""
        menu = self.menuBar().addMenu(title)
        for action in actions:
            if action is None:
                menu.addSeparator()
            else:
                menu.addAction(action)

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

        try:
            pixmap = QPixmap.fromImage(load_image(image_path))
        except ImageLoadError:
            pixmap = QPixmap()      # the Load page already checked it; the file may have gone since
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

        pairs = engine.get_angle_pairs()
        if pairs:
            metrics["Primary Cobb Angle"].set_value("{:.1f}°".format(engine.get_selected_cobb_angle()))
        else:
            metrics["Primary Cobb Angle"].set_value("No curve above 10°")

        apex_idx, deviation_px = engine.get_apex()
        self.workspace_page.show_curves([self._curve_text(curve) for curve in engine.curve_summaries()])
        if apex_idx is not None:
            metrics["Apex"].set_value(engine.get_level(apex_idx))
            metrics["CSVL Deviation"].set_value("{:.1f} px".format(deviation_px))
        else:
            metrics["Apex"].set_value("-")
            metrics["CSVL Deviation"].set_value("-")

        metrics["Vertebrae"].set_value(str(len(engine.get_detections())))
        self._update_window_title()

    @staticmethod
    def _curve_text(curve):
        """One panel line for a CurveSummary: the angle, then its levels and
        apex in small text. The whole-spine apex is in bold, as in the JPG."""
        details = "{}–{}".format(curve.upper_level, curve.lower_level)
        if curve.apex_level is not None:
            apex = curve.apex_level
            if curve.is_main_apex:
                apex = "<b style='color:{};'>{}</b>".format(theme.TEXT_PRIMARY, apex)
            details += " · apex " + apex
        return "{:.1f}°&nbsp;&nbsp;<span style='font-size:10pt; font-weight:400; color:{};'>{}</span>".format(
            curve.angle, theme.TEXT_MUTED, details)

    def _show_processing_time(self):
        """The time saved with the result ("-" for older projects)."""
        text = self.session.model_engine.processing_time_text()
        self.workspace_page.metrics["Processing Time"].set_value(text or "-")

    def _on_analysis_completed(self, elapsed):
        if self.session.model_engine is not None:
            self._show_processing_time()

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
            self._apply_curve_colors()
            self._preload_local_model()

    def _apply_curve_colors(self):
        """Use the saved curve colors on the image and in the panel."""
        colors = settings.curve_colors()
        self.overlay_layer.set_curve_colors(colors)
        self.workspace_page.set_curve_colors(colors)

    def _on_export_clicked(self):
        if self.session.model_engine is None:
            return
        image_item = self.workspace_page.canvas.image_item
        dialog = ExportDialog(
            self.session.model_engine, self,
            source_name=self._source_name(),
            image=image_item.pixmap() if image_item is not None else None,
            curve_colors=self.overlay_layer.curve_colors,
            edited=self.session.has_edits(),
        )
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
            self._show_processing_time()
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
