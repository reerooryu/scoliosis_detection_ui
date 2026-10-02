# Image viewer for the X-ray: pan, zoom and fit-to-view. It knows nothing
# about AI results. modules/overlay.py draws those as separate items on top,
# so the original pixels are never changed.

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


class ImageCanvas(QGraphicsView):
    """QGraphicsView/Scene viewer with pan, zoom, and fit-to-view behavior."""

    # Emitted whenever the zoom level changes (zoom in, zoom out, fit).
    view_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # View mode: pan with mouse
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setFrameStyle(QGraphicsView.NoFrame)
        self.setBackgroundBrush(Qt.black)

        self.image_item = None
        self.zoom_factor = 1.0
        # While True, resizing the window re-fits the image. Set to False
        # once the user zooms, so a resize does not undo their zoom.
        self._auto_fit = True

    def load_image(self, pixmap):
        """Sets the base image layer. The pixmap itself is never modified."""
        self._scene.clear()
        self.image_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self.image_item)
        self._scene.setSceneRect(self.image_item.boundingRect())
        self._auto_fit = True
        self.fit_in_view()

    def fit_in_view(self):
        """Fits the entire image inside the viewport, preserving aspect ratio."""
        if self.image_item is not None:
            self._auto_fit = True
            self.fitInView(self.image_item, Qt.KeepAspectRatio)
            self.zoom_factor = self.transform().m11()
            self.view_changed.emit()

    def zoom_in(self):
        self._auto_fit = False
        self.scale(1.15, 1.15)
        self.zoom_factor = self.transform().m11()
        self.view_changed.emit()

    def zoom_out(self):
        self._auto_fit = False
        self.scale(1.0 / 1.15, 1.0 / 1.15)
        self.zoom_factor = self.transform().m11()
        self.view_changed.emit()

    def wheelEvent(self, event):
        """CTRL+Scroll zooms; plain scroll is left to default (pan/scroll) behavior."""
        if event.modifiers() == Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event):
        """Re-fit on resize until the user zooms. Without the _auto_fit check,
        the scrollbars that appear on zoom-in trigger a resize that snaps the
        zoom straight back."""
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_in_view()

    def showEvent(self, event):
        """Ensures the image fits the view as soon as the canvas is first shown."""
        super().showEvent(event)
        if self._auto_fit:
            self.fit_in_view()
            QTimer.singleShot(100, self._delayed_initial_fit)

    def _delayed_initial_fit(self):
        """Fit once more shortly after the first show, in case the window
        size was not final yet. Skipped if the user has already zoomed."""
        if self._auto_fit:
            self.fit_in_view()
