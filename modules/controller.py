# AnalysisController: the workflow for one assessment -- submit / retry /
# reset, the background inference request, landmark edits (drag / undo /
# redo / reset edits) and project save / open.
#
# It changes an AnalysisSession (modules/session.py) and redraws the
# OverlayLayer (modules/overlay.py). Anything MainWindow needs to show
# (status text, error dialogs, processing time) is sent out as a signal.

import logging
import os

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QPixmap

from modules.model_mock import ScoliosisModelEngine
from modules.parser import InferenceWorker, validate_inference_payload
from modules.project import write_project, read_project, ProjectLoadError

logger = logging.getLogger(__name__)


class AnalysisController(QObject):
    """Runs the workflow for one AnalysisSession + OverlayLayer pair.

    api_url_provider is a function, not a fixed value, so each request uses
    the URL currently saved in Settings.
    """

    status_message = Signal(str)
    error_dialog = Signal(str, str)     # (title, message) -- window shows a QMessageBox
    busy_changed = Signal(bool)         # True while an inference request is in flight
    analysis_completed = Signal(float)  # elapsed seconds -- for the "Processing Time" metric

    def __init__(self, session, overlay_layer, api_url_provider, timeout, parent=None):
        super().__init__(parent)
        self.session = session
        self.overlay_layer = overlay_layer
        self._api_url_provider = api_url_provider
        self._timeout = timeout

        self._next_request_id = 0
        self._active_request_id = None
        # True while a request's result is still awaited.
        self._busy = False
        # request_id -> (QThread, InferenceWorker). Both are held until the
        # thread finishes so Qt does not delete them too early.
        self._inference_jobs = {}

    # ------------------------------------------------------------------
    # Submit / retry
    # ------------------------------------------------------------------

    def submit(self, image_path):
        """Start a fresh analysis for a newly loaded image. The overlay is
        cleared first: the caller's canvas.load_image() wipes the scene, and
        the overlay must not keep references to the deleted items."""
        self._cancel_inference_jobs()
        self.overlay_layer.clear()
        self.session.start_loading(image_path)
        self.status_message.emit(f"Loaded: {os.path.basename(image_path)} — running AI analysis…")
        self._run_inference(image_path)

    def retry(self):
        if not self.session.image_path:
            return
        if self._busy:
            # Do not stack duplicate requests on the server.
            self.status_message.emit("AI analysis is already running…")
            return
        self.status_message.emit("Retrying AI analysis…")
        self._run_inference(self.session.image_path)

    def _run_inference(self, image_path):
        api_url = self._api_url_provider()
        self._next_request_id += 1
        request_id = self._next_request_id
        self._active_request_id = request_id
        self._busy = True
        self.busy_changed.emit(True)

        thread = QThread(self)
        thread.setProperty("request_id", request_id)
        worker = InferenceWorker(request_id, image_path, api_url=api_url, timeout=self._timeout)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.succeeded.connect(self._on_inference_succeeded)
        worker.failed.connect(self._on_inference_failed)
        # finished fires after success or failure; Qt then stops the thread
        # and deletes both objects.
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_inference_thread_finished)

        self._inference_jobs[request_id] = (thread, worker)
        thread.start()

    def _on_inference_succeeded(self, request_id, data, elapsed):
        if request_id != self._active_request_id:
            return  # stale result from a superseded request (e.g. after Reset)
        self._busy = False
        self.busy_changed.emit(False)

        model_engine = ScoliosisModelEngine(autoload=False)
        model_engine.load_from_dict(data)
        # Match the result to the displayed image size. Skip this if the file
        # can no longer be read: a 0x0 size would move every landmark to (0, 0).
        pixmap = QPixmap(self.session.image_path)
        if not pixmap.isNull():
            model_engine.scale_coordinates(pixmap.width(), pixmap.height())
        # Take the "Reset Edits" baseline after scaling, not before.
        model_engine.capture_baseline()

        self.session.set_result(model_engine)

        self.overlay_layer.clear()
        self.overlay_layer.render(model_engine)

        count = len(model_engine.get_detections())
        self.status_message.emit(f"Analysis complete — {count} vertebrae detected.")
        self.analysis_completed.emit(elapsed)

    def _on_inference_failed(self, request_id, message):
        if request_id != self._active_request_id:
            return
        self._busy = False
        self.busy_changed.emit(False)
        self.session.set_error()
        self.status_message.emit("AI analysis unavailable — showing image only.")
        self.error_dialog.emit(
            "AI Analysis Unavailable",
            f"{message}\n\nYou can still view and zoom the image. "
            "Use View → Retry AI Analysis once the backend is reachable."
        )

    def _on_inference_thread_finished(self):
        """Release the Python references once Qt has stopped the thread."""
        thread = self.sender()
        request_id = thread.property("request_id") if thread is not None else None
        self._inference_jobs.pop(request_id, None)

    def _cancel_inference_jobs(self):
        """Mark every outstanding request as stale. A running HTTP call
        cannot be aborted, so it runs to its timeout and its result is
        ignored."""
        self._active_request_id = None
        self._busy = False
        for _thread, worker in self._inference_jobs.values():
            worker.cancel()

    def has_pending_jobs(self):
        return bool(self._inference_jobs)

    # ------------------------------------------------------------------
    # Landmark edits (Edit Mode drag / undo / redo / reset edits)
    # ------------------------------------------------------------------

    def on_drag_started(self, det_idx, kp_idx):
        """Mouse press on a handle: save one undo step before the change."""
        self.session.snapshot_for_undo()

    def on_keypoint_dragged(self, det_idx, kp_idx, x, y):
        """Every mouse-move tick of a drag: update the model and redraw."""
        self.session.apply_keypoint_drag(det_idx, kp_idx, x, y)
        if self.session.model_engine is not None:
            self.overlay_layer.render(self.session.model_engine)

    def on_drag_finished(self, det_idx, kp_idx):
        """Mouse release: refresh the undo/redo/"has edits" state."""
        self.session.refresh_edit_state()

    def undo(self):
        if not self.session.undo():
            return
        self.overlay_layer.clear()
        self.overlay_layer.render(self.session.model_engine)
        self.status_message.emit("Undid last landmark adjustment.")

    def redo(self):
        if not self.session.redo():
            return
        self.overlay_layer.clear()
        self.overlay_layer.render(self.session.model_engine)
        self.status_message.emit("Redid landmark adjustment.")

    def reset_edits(self):
        if not self.session.reset_edits():
            return
        self.overlay_layer.clear()
        self.overlay_layer.render(self.session.model_engine)
        self.status_message.emit("Manual edits reset to the AI's original result.")

    # ------------------------------------------------------------------
    # Full reset
    # ------------------------------------------------------------------

    def reset(self):
        self._cancel_inference_jobs()
        self.overlay_layer.clear()
        self.session.clear()
        self.status_message.emit("Load a spine X-ray image to begin.")

    # ------------------------------------------------------------------
    # Project save / open (modules/project.py)
    # ------------------------------------------------------------------

    def save_project(self, project_path):
        """Save the session to project_path. Returns True on success; on
        failure shows an error and leaves the session unchanged."""
        if self.session.model_engine is None:
            return False

        image_bytes = self.session.image_bytes
        image_ext = self.session.image_ext
        original_filename = self.session.original_filename

        if image_bytes is None:
            # First save after a fresh Submit: read the image once and keep
            # it in memory, so later saves do not need the original file.
            try:
                with open(self.session.image_path, "rb") as f:
                    image_bytes = f.read()
            except OSError as exc:
                self.error_dialog.emit(
                    "Save Project Failed", f"Could not read the source image: {exc}"
                )
                return False
            image_ext = os.path.splitext(self.session.image_path)[1] or ".png"
            original_filename = os.path.basename(self.session.image_path)
            self.session.image_bytes = image_bytes
            self.session.image_ext = image_ext
            self.session.original_filename = original_filename

        try:
            write_project(
                self.session.model_engine, image_bytes, image_ext,
                original_filename, project_path,
            )
        except (OSError, ValueError) as exc:
            self.error_dialog.emit("Save Project Failed", str(exc))
            return False

        self.session.mark_project_saved(project_path)
        self.status_message.emit(f"Project saved to {os.path.basename(project_path)}.")
        return True

    def open_project(self, project_path):
        """Open a saved .sdproj without contacting the server. Returns True
        on success; on failure shows an error and leaves the session unchanged.

        The overlay is not drawn here. Label placement needs the canvas to
        be visible and laid out, so MainWindow calls finish_open_project()
        shortly after switching to the workspace page.
        """
        try:
            image_bytes, image_ext, model_data, baseline_data, metadata = read_project(project_path)
        except ProjectLoadError as exc:
            self.error_dialog.emit("Could Not Open Project", str(exc))
            return False

        # Same checks as a fresh server result, so a damaged or hand-edited
        # project fails here with a message instead of part-way through loading.
        try:
            validate_inference_payload(model_data)
            validate_inference_payload(baseline_data)
        except ValueError as exc:
            self.error_dialog.emit(
                "Could Not Open Project",
                f"The project's detection data is not valid: {exc}"
            )
            return False

        pixmap = QPixmap()
        if not pixmap.loadFromData(image_bytes):
            self.error_dialog.emit(
                "Could Not Open Project",
                "The project's embedded image could not be decoded."
            )
            return False

        model_engine = ScoliosisModelEngine(autoload=False)
        model_engine.load_from_dict(model_data)
        model_engine.restore_baseline(baseline_data)

        self._cancel_inference_jobs()

        # Clear the overlay before the canvas wipes its scene (same as submit()).
        self.overlay_layer.clear()
        self.overlay_layer.canvas.load_image(pixmap)

        self.session.set_project_loaded(
            model_engine, image_bytes, image_ext,
            metadata.get("original_filename"), project_path,
        )

        count = len(model_engine.get_detections())
        self.status_message.emit(
            f"Opened project: {os.path.basename(project_path)} — {count} vertebrae detected."
        )
        return True

    def finish_open_project(self):
        """Draw the overlay for a project loaded by open_project(). Does
        nothing if the session was reset in the meantime."""
        if self.session.model_engine is not None:
            self.overlay_layer.render(self.session.model_engine)
