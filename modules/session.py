# AnalysisSession: the state for one loaded image. No widgets, no threads
# and no HTTP. Only AnalysisController (modules/controller.py) changes it;
# MainWindow just listens to its signals.

from PySide6.QtCore import QObject, Signal


class AnalysisSession(QObject):
    """State for one loaded image: its source, the AI result, and whether
    there are unexported or unsaved edits.

      state_changed(str)   -- one of the STATE_* values below
      metrics_changed()    -- the measurement panel should refresh
      edit_state_changed() -- undo/redo/"has edits" may have changed
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
        # The source image is kept in memory so Save Project still works if
        # the original file is moved or deleted.
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
        """Like set_result(), for a reopened project: remembers the project
        path so "Save Project" writes back to the same file."""
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
    # Landmark edits. The model engine owns the undo history; these keep
    # the dirty flags and signals in step with it.
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
        """Re-emit edit_state_changed. Called when a drag ends, because
        nothing re-checks has_edits() during the drag."""
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
