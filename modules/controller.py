# AnalysisController: the workflow/use-case layer for one clinical
# assessment -- submit / retry / reset, the background inference-request
# lifecycle, landmark-edit operations (drag / undo / redo / reset edits),
# and project save/open (modules/project.py) so a clinician can resume an
# in-progress assessment later without re-running AI inference. This used
# to all live directly on MainWindow, alongside its menu/toolbar/dialog
# construction; it's pulled out here so MainWindow can go back to just
# being a composition root + navigation owner + dialog factory, and bind
# its widgets to signals instead of being handed explicit "go refresh
# yourself now" calls after every operation. See AGENTS.md.
#
# This controller mutates an AnalysisSession (modules/session.py) and drives
# an OverlayLayer (modules/overlay.py) directly -- the overlay is really "the
# view of the model," not a plain widget the window can bind generically, so
# it's reasonable for the use-case layer to push updates to it directly.
# Everything the window itself needs (status text, error dialogs, the
# "processing time" figure) comes back out through signals below.

import logging
import os

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QPixmap

from modules.model_mock import ScoliosisModelEngine
from modules.parser import InferenceWorker
from modules.project import write_project, read_project, ProjectLoadError

logger = logging.getLogger(__name__)


class AnalysisController(QObject):
    """Orchestrates one AnalysisSession + OverlayLayer pair.

    api_url_provider is a zero-arg callable (SettingsDialog.get_saved_api_url)
    rather than a captured value, so each request picks up whatever's
    currently saved in Settings without the controller needing to be told
    about settings changes.
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
        # Request ID -> (QThread, InferenceWorker). Keeping both references
        # until QThread.finished prevents premature Python/C++ destruction.
        self._inference_jobs = {}

    # ------------------------------------------------------------------
    # Submit / retry
    # ------------------------------------------------------------------

    def submit(self, image_path):
        """Starts a fresh analysis for a newly loaded image. Cancels any
        outstanding request and clears the overlay before the caller loads
        the new pixmap into the canvas -- ImageCanvas.load_image() calls
        scene.clear(), which deletes the overlay's C++ items, so tracked
        Python references need to be dropped first."""
        self._cancel_inference_jobs()
        self.overlay_layer.clear()
        self.session.start_loading(image_path)
        self.status_message.emit(f"Loaded: {os.path.basename(image_path)} — running AI analysis…")
        self._run_inference(image_path)

    def retry(self):
        if self.session.image_path:
            self.status_message.emit("Retrying AI analysis…")
            self._run_inference(self.session.image_path)

    def _run_inference(self, image_path):
        api_url = self._api_url_provider()
        self._next_request_id += 1
        request_id = self._next_request_id
        self._active_request_id = request_id
        self.busy_changed.emit(True)

        thread = QThread(self)
        thread.setProperty("request_id", request_id)
        worker = InferenceWorker(request_id, image_path, api_url=api_url, timeout=self._timeout)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.succeeded.connect(self._on_inference_succeeded)
        worker.failed.connect(self._on_inference_failed)
        # The worker emits finished after either outcome. The standard Qt
        # lifecycle chain then stops and deletes both C++ objects safely.
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_inference_thread_finished)

        self._inference_jobs[request_id] = (thread, worker)
        thread.start()

    def _on_inference_succeeded(self, request_id, data, elapsed):
        if request_id != self._active_request_id:
            return  # stale result from a superseded request (e.g. after Reset)
        self.busy_changed.emit(False)

        pixmap = QPixmap(self.session.image_path)
        model_engine = ScoliosisModelEngine(autoload=False)
        model_engine.load_from_dict(data)
        model_engine.scale_coordinates(pixmap.width(), pixmap.height())
        # Capture the AI's original result *after* scaling, so "Reset Edits"
        # restores coordinates in the same space the canvas actually displays.
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
        """Invalidate every outstanding request without destroying threads.

        ``requests`` has no safe cross-thread abort API, so a running HTTP
        call is allowed to reach its configured timeout. Its worker then
        exits normally and the lifecycle connections above clean it up.
        """
        self._active_request_id = None
        for _thread, worker in self._inference_jobs.values():
            worker.cancel()

    def has_pending_jobs(self):
        return bool(self._inference_jobs)

    # ------------------------------------------------------------------
    # Landmark edits (Edit Mode drag / undo / redo / reset edits)
    # ------------------------------------------------------------------

    def on_drag_started(self, det_idx, kp_idx):
        """Fired once per drag gesture (mouse press on a handle) -- this is
        the undo checkpoint, capturing the state right before this specific
        adjustment begins."""
        self.session.snapshot_for_undo()

    def on_keypoint_dragged(self, det_idx, kp_idx, x, y):
        """Fired continuously while a handle is being dragged (every
        mouse-move tick). See OverlayLayer.render()/_render_cobb_overlays()
        for why this doesn't tear down and rebuild the whole overlay on
        every tick."""
        self.session.apply_keypoint_drag(det_idx, kp_idx, x, y)
        if self.session.model_engine is not None:
            self.overlay_layer.render(self.session.model_engine)

    def on_drag_finished(self, det_idx, kp_idx):
        """Fired once per drag gesture (mouse release). The live updates
        during the drag already kept everything consistent, but has_edits()
        is computed lazily from the model and nothing else re-checks it
        mid-drag, so this is the point to refresh it."""
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
    # Project save / open (modules/project.py) -- resuming an in-progress
    # assessment later, without re-running AI inference.
    # ------------------------------------------------------------------

    def save_project(self, project_path):
        """Writes the current session + model_engine state to project_path.
        Returns True on success; emits error_dialog and returns False on
        failure (nothing in the session changes in that case)."""
        if self.session.model_engine is None:
            return False

        image_bytes = self.session.image_bytes
        image_ext = self.session.image_ext
        original_filename = self.session.original_filename

        if image_bytes is None:
            # A session from a fresh Submit only has a live file path until
            # the first save -- read and cache the bytes here so every
            # subsequent save (and any save after the original file might
            # move or be deleted) no longer depends on that on-disk path.
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
        """Loads a previously-saved .sdproj bundle: decodes the embedded
        image, restores the model engine's current + baseline data, and
        loads the image into the canvas -- all without contacting the
        inference backend. Returns True on success; emits error_dialog and
        returns False on failure (nothing in the session changes in that
        case).

        Deliberately does NOT render the overlay yet -- see
        finish_open_project(). OverlayLayer's Cobb-label placement maps
        scene coordinates to screen coordinates using the canvas's current
        viewport size/transform, which is only correct once the canvas has
        actually been shown and Qt has finished laying it out. In the
        Submit flow this is never a problem by accident: the overlay isn't
        rendered until the AI result comes back over the network, and that
        round-trip is more than enough time for the canvas (already loaded
        with the raw image, and about to be switched into view) to settle.
        open_project() has no such gap -- if the workspace page has never
        been shown yet in this run of the app, rendering synchronously here
        would compute label positions against a stale/unsettled viewport
        and place them "way off" the image (only self-correcting the next
        time the page is shown, once its geometry is already settled).
        MainWindow.open_project() calls finish_open_project() only after
        switching to the workspace page and letting Qt process the
        resulting show/resize events.
        """
        try:
            image_bytes, image_ext, model_data, baseline_data, metadata = read_project(project_path)
        except ProjectLoadError as exc:
            self.error_dialog.emit("Could Not Open Project", str(exc))
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

        # Clear the overlay before the canvas swaps images, same ordering
        # requirement as submit(): ImageCanvas.load_image() calls
        # scene.clear(), which deletes the overlay's C++ items out from
        # under any still-tracked Python references.
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
        """Renders the overlay for a project loaded by open_project(), once
        the caller (MainWindow) has made the workspace page visible and let
        Qt settle its layout. No-op if nothing is loaded (e.g. the user hit
        Full Reset in the brief window before this was called)."""
        if self.session.model_engine is not None:
            self.overlay_layer.render(self.session.model_engine)
