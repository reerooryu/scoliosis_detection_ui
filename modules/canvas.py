# Image viewer for the X-ray: pan, zoom and fit-to-view. It knows nothing
# about AI results. modules/overlay.py draws those as separate items on top,
# so the original pixels are never changed.
#
# Controls:
#   mouse wheel                -> zoom in/out at the cursor
#   trackpad pinch             -> zoom in/out at the cursor
#   trackpad two-finger scroll -> move the image
#   click and drag             -> move the image (when zoomed in)
#   Ctrl/Cmd + scroll          -> zoom, on any device

from PySide6.QtCore import Qt, QTimer, Signal, QEvent
from PySide6.QtGui import QPainter, QInputDevice
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem

# Zoom limits: out to half the fit-to-view size, in to 16 screen pixels per
# image pixel.
MIN_ZOOM_OF_FIT = 0.5
MAX_ZOOM = 16.0
# One mouse-wheel notch (or one toolbar click) zooms by this factor.
ZOOM_STEP = 1.15


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
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # click and drag moves the image
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setFrameStyle(QGraphicsView.NoFrame)
        self.setBackgroundBrush(Qt.black)

        self.image_item = None
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
            self.view_changed.emit()

    def zoom_in(self):
        self._zoom_by(ZOOM_STEP)

    def zoom_out(self):
        self._zoom_by(1.0 / ZOOM_STEP)

    def _fit_scale(self):
        """The zoom level at which the whole image fits the viewport."""
        rect = self.image_item.boundingRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return 1.0
        viewport = self.viewport().rect()
        return min(viewport.width() / rect.width(), viewport.height() / rect.height())

    def _zoom_by(self, factor):
        """Zoom by factor, kept within the zoom limits. The point under the
        cursor stays put (AnchorUnderMouse); from the toolbar or a shortcut,
        the view center stays put."""
        if self.image_item is None or factor <= 0:
            return
        current = self.transform().m11()
        if current <= 0:
            return
        low = min(self._fit_scale() * MIN_ZOOM_OF_FIT, current)
        high = max(MAX_ZOOM, current)
        target = max(low, min(current * factor, high))
        if abs(target - current) < 1e-9:
            return
        self._auto_fit = False
        self.scale(target / current, target / current)
        self.view_changed.emit()

    @staticmethod
    def _is_mouse_wheel(event):
        """True for a mouse wheel, False for a trackpad. Trackpad scrolls come
        with scroll phases (begin / update / end) on macOS, or are reported
        as a touchpad device."""
        if event.phase() != Qt.NoScrollPhase:
            return False
        device = event.device()
        if device is not None and device.type() == QInputDevice.DeviceType.TouchPad:
            return False
        return True

    def wheelEvent(self, event):
        """Mouse wheel and Ctrl/Cmd + scroll zoom; trackpad scrolling moves
        the image."""
        delta = event.angleDelta().y()
        ctrl = bool(event.modifiers() & Qt.ControlModifier)
        plain_mouse_wheel = self._is_mouse_wheel(event) and event.modifiers() == Qt.NoModifier
        if delta and (ctrl or plain_mouse_wheel):
            # 120 = one wheel notch. Smaller steps (smooth wheels, Windows
            # trackpad pinch) zoom proportionally less.
            self._zoom_by(ZOOM_STEP ** (delta / 120.0))
            event.accept()
            return
        super().wheelEvent(event)

    def viewportEvent(self, event):
        """Trackpad pinch arrives as a native zoom gesture (macOS, and Linux
        with libinput). On Windows it arrives as Ctrl + scroll instead."""
        if event.type() == QEvent.NativeGesture and event.gestureType() == Qt.ZoomNativeGesture:
            self._zoom_by(1.0 + event.value())
            event.accept()
            return True
        return super().viewportEvent(event)

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
