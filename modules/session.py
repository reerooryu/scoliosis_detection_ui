# AnalysisSession: the per-loaded-image clinical state, separated out of
# MainWindow so the window can bind to signals instead of being told to
# manually refresh every widget after every operation.
#
# This holds *state* only -- it knows nothing about Qt widgets, threads, or
# the inference HTTP call. modules/controller.py:AnalysisController is the
# only thing that mutates a session; modules/main_window.py just listens.
# See AGENTS.md for the full rationale behind this split.

from PySide6.QtCore import QObject, Signal


class AnalysisSession(QObject):
    """Holds the state for one loaded image: its path, the AI result (if
    any), and whether it has unexported manual edits.

    Three separate signals rather than one generic "changed", since most
    listeners only care about one of these -- e.g. the toolbar's undo/redo
    buttons don't need to re-check on every keypoint-drag tick, only when
    undo/redo availability itself actually changes:

      state_changed(str)   -- lifecycle: one of the STATE_* constants below
      metrics_changed()    -- the measurement panel should re-read model_engine
      edit_state_changed() -- undo/redo/"has edits" availability may differ
    """

    state_changed = Signal(str)
    metrics_changed = Signal()
    edit_state_changed = Signal()

    STATE_EMPTY = "empty"      # no image loaded
    STATE_LOADING = "loading"  # image loaded, inference request in flight
    STATE_READY = "ready"      # a result is loaded and displayed
    STATE_ERROR = "error"      # the last inference request failed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        # Raw source-image bytes, ext, and original basename -- kept in
        # memory (not just referenced by image_path) so Save Project never
        # depends on the original file still existing on disk. Populated
        # either lazily on first save (from a live Submit) or immediately
        # on opening a project (see AnalysisController.save_project() /
        # open_project()).
        self.image_bytes = None
        self.image_ext = None
        self.original_filename = None
        self.model_engine = None
        self.dirty = False  # True once landmarks have been dragged since the last export
        self.project_path = None    # path of the currently open .sdproj, if any
        self.project_dirty = False  # True once changed since the last Save Project
        self.state = self.STATE_EMPTY

    @property
    def has_result(self):
        return self.model_engine is not None

    def has_project(self):
        return self.project_path is not None

    def _set_state(self, state):
        self.state = state
        self.state_changed.emit(state)

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def start_loading(self, image_path):
        self.image_path = image_path
        self.image_bytes = None
        self.image_ext = None
        self.original_filename = None
        self.model_engine = None
        self.dirty = False
        self.project_path = None
        self.project_dirty = False
        self._set_state(self.STATE_LOADING)

    def set_result(self, model_engine, image_bytes=None, image_ext=None, original_filename=None):
        self.model_engine = model_engine
        if image_bytes is not None:
            self.image_bytes = image_bytes
            self.image_ext = image_ext
            self.original_filename = original_filename
        self.dirty = False
        self._set_state(self.STATE_READY)
        self.metrics_changed.emit()
        self.edit_state_changed.emit()

    def set_project_loaded(self, model_engine, image_bytes, image_ext, original_filename, project_path):
        """Like set_result(), but for reopening a saved project: also seeds
        project_path so subsequent "Save Project" (not "Save Project As...")
        writes back to the same file, and marks it not-yet-re-modified."""
        self.image_path = None
        self.project_path = project_path
        self.project_dirty = False
        self.set_result(model_engine, image_bytes, image_ext, original_filename)

    def mark_project_saved(self, project_path):
        self.project_path = project_path
        self.project_dirty = False

    def set_error(self):
        self._set_state(self.STATE_ERROR)

    def clear(self):
        self.image_path = None
        self.image_bytes = None
        self.image_ext = None
        self.original_filename = None
        self.model_engine = None
        self.dirty = False
        self.project_path = None
        self.project_dirty = False
        self._set_state(self.STATE_EMPTY)
        self.edit_state_changed.emit()

    # ------------------------------------------------------------------
    # Landmark edits (undo/redo/reset delegate to ScoliosisModelEngine,
    # which owns the actual history stacks -- this just keeps `dirty` and
    # the signal emissions in sync with it)
    # ------------------------------------------------------------------

    def apply_keypoint_drag(self, det_idx, kp_idx, x, y):
        if self.model_engine is None:
            return
        self.model_engine.update_keypoint(det_idx, kp_idx, x, y)
        self.dirty = True
        self.project_dirty = True
        self.metrics_changed.emit()

    def snapshot_for_undo(self):
        if self.model_engine is None:
            return
        self.model_engine.snapshot_for_undo()
        self.edit_state_changed.emit()

    def refresh_edit_state(self):
        """Re-emit edit_state_changed without altering anything -- used once
        a drag gesture ends, to catch has_edits() becoming true partway
        through (it's computed lazily from the model, so nothing else
        re-checks it mid-drag)."""
        self.edit_state_changed.emit()

    def undo(self):
        if self.model_engine is None or not self.model_engine.undo():
            return False
        self.dirty = self.model_engine.has_edits()
        self.project_dirty = True
        self.metrics_changed.emit()
        self.edit_state_changed.emit()
        return True

    def redo(self):
        if self.model_engine is None or not self.model_engine.redo():
            return False
        self.dirty = self.model_engine.has_edits()
        self.project_dirty = True
        self.metrics_changed.emit()
        self.edit_state_changed.emit()
        return True

    def reset_edits(self):
        if self.model_engine is None or not self.model_engine.has_edits():
            return False
        self.model_engine.reset_edits()
        self.dirty = False
        self.project_dirty = True
        self.metrics_changed.emit()
        self.edit_state_changed.emit()
        return True

    def can_undo(self):
        return self.has_result and self.model_engine.can_undo()

    def can_redo(self):
        return self.has_result and self.model_engine.can_redo()

    def has_edits(self):
        return self.has_result and self.model_engine.has_edits()
