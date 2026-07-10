# Pure image-display canvas built on the Qt Graphics View Framework.
#
# Renders the original X-ray as a single base-layer QGraphicsPixmapItem.
# This module intentionally has no knowledge of AI predictions or
# measurement overlays -- those are layered on top by modules/overlay.py
# once the landmark / Cobb-angle editing features are built. Keeping the
# base viewer separate means the original pixel data is never touched by
# anything drawn on top of it.

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


class ImageCanvas(QGraphicsView):
    """QGraphicsView/Scene viewer with pan, zoom, and fit-to-view behavior."""

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
        # True until the user manually zooms/wheels; while True, any resize
        # (including the initial show) keeps re-fitting the image. Once the
        # user zooms in/out, resizes must stop overriding their chosen zoom
        # level -- see resizeEvent for why this matters.
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

    def zoom_in(self):
        self._auto_fit = False
        self.scale(1.15, 1.15)
        self.zoom_factor = self.transform().m11()

    def zoom_out(self):
        self._auto_fit = False
        self.scale(1.0 / 1.15, 1.0 / 1.15)
        self.zoom_factor = self.transform().m11()

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
        """Keeps the image fitted to the viewport on resize -- but only until
        the user has manually zoomed. Without the _auto_fit guard, zooming in
        past the viewport's size makes scrollbars appear, which shrinks the
        viewport and fires a resizeEvent right back into fit_in_view(),
        silently snapping the zoom back to fit level on every zoom-in click
        past that point (this was reported as "zoom in doesn't work past the
        image's fit size")."""
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
        """One-time catch-up fit ~100ms after first show, in case the
        window's real on-screen size wasn't settled yet at showEvent time.
        Re-checks _auto_fit at fire time rather than just when it was
        scheduled: if the user manually zoomed in during that window, this
        would otherwise silently snap their zoom back to fit a moment
        later."""
        if self._auto_fit:
            self.fit_in_view()
