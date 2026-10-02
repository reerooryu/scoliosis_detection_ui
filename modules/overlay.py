# Overlay items drawn on top of the X-ray: vertebra outlines, landmark
# handles, Cobb lines and labels, and the CSVL. OverlayLayer (bottom of this
# file) creates and updates them from a ScoliosisModelEngine. The X-ray
# itself is never modified.

import math

import shiboken6

from PySide6.QtCore import QPointF, QPoint, Signal, QObject, Qt
from PySide6.QtGui import QPen, QBrush, QColor, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsPolygonItem, QGraphicsLineItem,
    QGraphicsTextItem
)

from config import (
    KP_CENTER, KP_TOP_LEFT, KP_TOP_RIGHT, KP_BOTTOM_LEFT, KP_BOTTOM_RIGHT,
    COLOR_VERTEBRA_OUTLINE, COLOR_CORRIDOR_FILL, COLOR_KEYPOINT_CENTER,
    COLOR_KEYPOINT_CORNER, COLOR_COBB_LINE, COLOR_CSVL_LINE,
    HANDLE_RADIUS, ACTIVE_HANDLE_RADIUS
)


class OverlaySignals(QObject):
    """Signals for landmark drags in Edit mode."""
    # Every mouse-move tick of a drag: (det_idx, kp_idx, x, y).
    keypoint_moved = Signal(int, int, float, float)
    # Once per drag, on mouse press and on release: (det_idx, kp_idx).
    drag_started = Signal(int, int)
    drag_finished = Signal(int, int)


class LandmarkHandleItem(QGraphicsEllipseItem):
    """Circle handle for one vertebra keypoint.

    The four corners are draggable. The center is shown but not draggable:
    it is always recalculated as the average of the corners.
    """

    def __init__(self, det_idx, kp_idx, x, y, parent_item=None, signals=None):
        super().__init__(-HANDLE_RADIUS, -HANDLE_RADIUS, HANDLE_RADIUS * 2, HANDLE_RADIUS * 2, parent_item)
        self.det_idx = det_idx
        self.kp_idx = kp_idx
        self.signals = signals
        self.is_dragging = False
        self.draggable = kp_idx != KP_CENTER

        self.setPen(QPen(QColor(0, 0, 0, 180), 1))
        if kp_idx == KP_CENTER:
            self.setBrush(QBrush(QColor(*COLOR_KEYPOINT_CENTER)))
        else:
            self.setBrush(QBrush(QColor(*COLOR_KEYPOINT_CORNER)))

        # Keep the handle the same on-screen size at any zoom level.
        flags = QGraphicsEllipseItem.ItemIgnoresTransformations
        if self.draggable:
            flags |= QGraphicsEllipseItem.ItemIsMovable | QGraphicsEllipseItem.ItemSendsGeometryChanges
        self.setFlags(flags)
        self.setAcceptHoverEvents(self.draggable)
        self.setPos(x, y)

    def hoverEnterEvent(self, event):
        self.setRect(-ACTIVE_HANDLE_RADIUS, -ACTIVE_HANDLE_RADIUS, ACTIVE_HANDLE_RADIUS * 2, ACTIVE_HANDLE_RADIUS * 2)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if not self.is_dragging:
            self.setRect(-HANDLE_RADIUS, -HANDLE_RADIUS, HANDLE_RADIUS * 2, HANDLE_RADIUS * 2)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if not self.draggable:
            event.ignore()
            return
        self.is_dragging = True
        if self.signals:
            self.signals.drag_started.emit(self.det_idx, self.kp_idx)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if not self.draggable:
            event.ignore()
            return
        self.is_dragging = False
        self.setRect(-HANDLE_RADIUS, -HANDLE_RADIUS, HANDLE_RADIUS * 2, HANDLE_RADIUS * 2)
        super().mouseReleaseEvent(event)
        if self.signals:
            self.signals.drag_finished.emit(self.det_idx, self.kp_idx)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange and self.is_dragging:
            if self.signals:
                self.signals.keypoint_moved.emit(self.det_idx, self.kp_idx, value.x(), value.y())
        return super().itemChange(change, value)


class VertebraOutlineItem(QGraphicsPolygonItem):
    """Polygon outlining the 4 corners of a single vertebra."""

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self.setPen(QPen(QColor(*COLOR_VERTEBRA_OUTLINE), 1.5))
        self.setBrush(QBrush(QColor(0, 120, 215, 15)))

    def update_polygon(self, keypoints):
        if len(keypoints) >= 5:
            poly = QPolygonF()
            poly.append(QPointF(keypoints[KP_TOP_LEFT][0], keypoints[KP_TOP_LEFT][1]))
            poly.append(QPointF(keypoints[KP_TOP_RIGHT][0], keypoints[KP_TOP_RIGHT][1]))
            poly.append(QPointF(keypoints[KP_BOTTOM_RIGHT][0], keypoints[KP_BOTTOM_RIGHT][1]))
            poly.append(QPointF(keypoints[KP_BOTTOM_LEFT][0], keypoints[KP_BOTTOM_LEFT][1]))
            self.setPolygon(poly)


class SpinalCorridorItem(QGraphicsPolygonItem):
    """Semi-transparent overlay covering the spinal canal corridor."""

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self.setPen(QPen(QColor(0, 204, 150, 100), 1))
        self.setBrush(QBrush(QColor(*COLOR_CORRIDOR_FILL)))

    def update_corridor(self, detections):
        if not detections:
            return
        poly = QPolygonF()
        for det in detections:
            kps = det.get("keypoints", [])
            if len(kps) >= 5:
                poly.append(QPointF(kps[KP_TOP_LEFT][0], kps[KP_TOP_LEFT][1]))
                poly.append(QPointF(kps[KP_BOTTOM_LEFT][0], kps[KP_BOTTOM_LEFT][1]))
        for det in reversed(detections):
            kps = det.get("keypoints", [])
            if len(kps) >= 5:
                poly.append(QPointF(kps[KP_BOTTOM_RIGHT][0], kps[KP_BOTTOM_RIGHT][1]))
                poly.append(QPointF(kps[KP_TOP_RIGHT][0], kps[KP_TOP_RIGHT][1]))
        self.setPolygon(poly)


class CobbMeasurementLineItem(QGraphicsLineItem):
    """Measurement line for a Cobb angle, drawn along a vertebra endplate."""

    def __init__(self, color=COLOR_COBB_LINE, parent_item=None):
        super().__init__(parent_item)
        self.set_color(color)

    def set_color(self, color):
        """Update only the pen color, preserving the measurement line width."""
        qcolor = QColor(*color) if isinstance(color, tuple) else QColor(color)
        pen = self.pen()
        pen.setColor(qcolor)
        pen.setWidth(2)
        self.setPen(pen)


class CSVLLineItem(QGraphicsLineItem):
    """Central Sacral Vertical Line reference overlay (dash-dot, magenta)."""

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        pen = QPen(QColor(*COLOR_CSVL_LINE), 2, Qt.DashDotLine)
        self.setPen(pen)


def extended_line_points(p1, p2, extend_len=200):
    """Returns (start, end) points for the p1->p2 line extended at both ends."""
    dx = p2.x() - p1.x()
    dy = p2.y() - p1.y()
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return None, None
    ux, uy = dx / length, dy / length
    start_p = QPointF(p1.x() - ux * extend_len, p1.y() - uy * extend_len)
    end_p = QPointF(p2.x() + ux * (extend_len + 100), p2.y() + uy * (extend_len + 100))
    return start_p, end_p


class OverlayLayer:
    """Creates and updates the overlay items for one ImageCanvas from a
    ScoliosisModelEngine's data."""

    def __init__(self, canvas, cobb_line_color=COLOR_COBB_LINE):
        self.canvas = canvas
        self.signals = OverlaySignals()
        self.cobb_line_color = cobb_line_color
        self.corridor_item = None
        self.outline_items = []
        self.handle_items = []
        self.cobb_lines = []
        self.cobb_texts = []
        self.csvl_item = None
        # (mid_y, min_x, max_x, label) per curve, kept so the labels can be
        # re-placed when the zoom changes.
        self._label_anchors = []
        self.interactive_mode = False

    def clear(self):
        """Remove all overlay items. QGraphicsScene.clear() may already have
        deleted them, so each one is checked before it is touched."""
        scene = self.canvas.scene()
        for item in self.outline_items + self.handle_items + self.cobb_lines + self.cobb_texts:
            self._remove_item_if_valid(scene, item)
        self._remove_item_if_valid(scene, self.corridor_item)
        self._remove_item_if_valid(scene, self.csvl_item)
        self.corridor_item = None
        self.outline_items = []
        self.handle_items = []
        self.cobb_lines = []
        self.cobb_texts = []
        self.csvl_item = None
        self._label_anchors = []

    @staticmethod
    def _remove_item_if_valid(scene, item):
        """Remove item from the scene only if Qt has not already deleted it."""
        if item is not None and shiboken6.isValid(item) and item.scene() is scene:
            scene.removeItem(item)

    def set_interactive(self, enabled):
        """Toggles landmark handle visibility (View mode vs Edit mode)."""
        self.interactive_mode = enabled
        for handle in self.handle_items:
            handle.setVisible(enabled)

    def set_cobb_line_color(self, color):
        """Apply a saved measurement-line color to current and future items."""
        self.cobb_line_color = color
        for line in self.cobb_lines:
            if shiboken6.isValid(line):
                line.set_color(color)

    def render(self, model_engine):
        """Create the overlay items on the first call; update them in place
        on later calls (for example on every tick of a drag)."""
        scene = self.canvas.scene()
        detections = model_engine.get_detections()
        angle_pairs = model_engine.get_angle_pairs()

        if self.corridor_item is None:
            self.corridor_item = SpinalCorridorItem()
            self.corridor_item.setZValue(1)
            scene.addItem(self.corridor_item)
        self.corridor_item.update_corridor(detections)

        if not self.outline_items:
            for d_idx, det in enumerate(detections):
                outline = VertebraOutlineItem()
                outline.setZValue(2)
                outline.update_polygon(det["keypoints"])
                scene.addItem(outline)
                self.outline_items.append(outline)

                for kp_idx, kp in enumerate(det["keypoints"]):
                    handle = LandmarkHandleItem(d_idx, kp_idx, kp[0], kp[1], signals=self.signals)
                    handle.setZValue(3)
                    handle.setVisible(self.interactive_mode)
                    scene.addItem(handle)
                    self.handle_items.append(handle)
        else:
            for idx, det in enumerate(detections):
                self.outline_items[idx].update_polygon(det["keypoints"])
            handle_idx = 0
            for det in detections:
                for kp in det["keypoints"]:
                    handle = self.handle_items[handle_idx]
                    # Do not move the handle being dragged: Qt is already
                    # moving it, and calling setPos() on it here can freeze
                    # or crash the app.
                    if not handle.is_dragging:
                        handle.setPos(kp[0], kp[1])
                    handle_idx += 1

        self._render_csvl(model_engine)
        self._render_cobb_overlays(detections, angle_pairs)

    def _render_csvl(self, model_engine):
        csvl_x = model_engine.get_csvl_x()
        if csvl_x is None:
            return
        image_item = getattr(self.canvas, "image_item", None)
        height = image_item.boundingRect().height() if image_item is not None else 2000

        if self.csvl_item is None:
            self.csvl_item = CSVLLineItem()
            self.csvl_item.setZValue(2.2)  # Above corridor/outlines, below Cobb lines/handles
            self.canvas.scene().addItem(self.csvl_item)
        self.csvl_item.setLine(csvl_x, 0, csvl_x, height)

    def _render_cobb_overlays(self, detections, angle_pairs):
        """Update the Cobb lines and labels in place. They are rebuilt only
        when the number of curves changes: creating and deleting scene items
        on every drag tick is unsafe in Qt and can freeze or crash the app."""
        scene = self.canvas.scene()
        expected_lines = len(angle_pairs) * 2
        expected_texts = len(angle_pairs)

        if len(self.cobb_lines) != expected_lines or len(self.cobb_texts) != expected_texts:
            for item in self.cobb_lines:
                self._remove_item_if_valid(scene, item)
            for item in self.cobb_texts:
                self._remove_item_if_valid(scene, item)

            self.cobb_lines = []
            for _ in range(expected_lines):
                line = CobbMeasurementLineItem(self.cobb_line_color)
                line.setZValue(2.5)
                scene.addItem(line)
                self.cobb_lines.append(line)

            self.cobb_texts = []
            for _ in range(expected_texts):
                txt = QGraphicsTextItem()
                txt.setZValue(4)
                # Keep the label the same on-screen size at any zoom level.
                txt.setFlag(QGraphicsTextItem.ItemIgnoresTransformations, True)
                scene.addItem(txt)
                self.cobb_texts.append(txt)

        # (mid_y, local_min_x, local_max_x, txt_item) per curve, resolved to
        # final screen positions after this loop by _place_labels_without_overlap.
        pending_labels = []
        for idx, pair in enumerate(angle_pairs):
            u_idx = pair.get("upper_detection_index")
            l_idx = pair.get("lower_detection_index")
            cobb_val = pair.get("cobb_angle", 0.0)
            if u_idx is None or l_idx is None:
                continue
            if not (0 <= u_idx < len(detections) and 0 <= l_idx < len(detections)):
                continue

            u_kps = detections[u_idx]["keypoints"]
            l_kps = detections[l_idx]["keypoints"]

            p1_u = QPointF(u_kps[KP_TOP_LEFT][0], u_kps[KP_TOP_LEFT][1])
            p2_u = QPointF(u_kps[KP_TOP_RIGHT][0], u_kps[KP_TOP_RIGHT][1])
            self._set_extended_line(self.cobb_lines[idx * 2], p1_u, p2_u)

            p1_l = QPointF(l_kps[KP_BOTTOM_LEFT][0], l_kps[KP_BOTTOM_LEFT][1])
            p2_l = QPointF(l_kps[KP_BOTTOM_RIGHT][0], l_kps[KP_BOTTOM_RIGHT][1])
            self._set_extended_line(self.cobb_lines[idx * 2 + 1], p1_l, p2_l)

            # Horizontal extent of just the vertebrae in this curve, so the
            # label can sit beside them.
            lo, hi = sorted((u_idx, l_idx))
            local_xs = [
                kp[0]
                for det in detections[lo:hi + 1]
                for kp in det["keypoints"]
            ]
            if local_xs:
                local_min_x, local_max_x = min(local_xs), max(local_xs)
            else:
                mid_x = (p1_u.x() + p1_l.x()) / 2.0
                local_min_x = local_max_x = mid_x

            mid_y = (p1_u.y() + p1_l.y()) / 2.0
            txt_item = self.cobb_texts[idx]
            txt_item.setHtml(
                "<div style='background-color: rgba(255,255,255,210); "
                "border: 1.5px solid rgb(255,87,34); padding:4px; border-radius:4px;'>"
                f"<b style='color: rgb(255,87,34); font-size:12px;'>Curve #{idx + 1}</b><br/>"
                f"<span style='color:#333; font-size:14px; font-weight:bold;'>Angle: {cobb_val:.2f}°</span>"
                "</div>"
            )
            pending_labels.append((mid_y, local_min_x, local_max_x, txt_item))

        self._label_anchors = pending_labels
        self._place_labels_without_overlap(pending_labels)

    def reposition_labels(self):
        """Re-place the Cobb labels for the current zoom. Their positions are
        worked out in screen pixels, so they go stale when the zoom changes."""
        if self._label_anchors and all(shiboken6.isValid(a[3]) for a in self._label_anchors):
            self._place_labels_without_overlap(self._label_anchors)

    def _place_labels_without_overlap(self, pending_labels, min_gap_px=10, hug_gap_px=16, edge_margin_px=6):
        """Place each Cobb label beside its own curve, on whichever side has
        more free space in the viewport. Labels on the same side are pushed
        down if they would overlap, then kept inside the viewport."""
        view = self.canvas
        viewport_w = view.viewport().width()
        last_bottom = {"left": None, "right": None}

        for mid_y, local_min_x, local_max_x, txt_item in pending_labels:
            left_edge_screen = view.mapFromScene(QPointF(local_min_x, mid_y)).x()
            right_edge_screen = view.mapFromScene(QPointF(local_max_x, mid_y)).x()
            rect = txt_item.boundingRect()
            label_w = rect.width()
            label_h = rect.height()

            left_space = left_edge_screen
            right_space = viewport_w - right_edge_screen

            if right_space >= left_space:
                side = "right"
                x = right_edge_screen + hug_gap_px
            else:
                side = "left"
                x = left_edge_screen - hug_gap_px - label_w

            # Keep the label on screen even if neither side has room.
            x = max(edge_margin_px, min(x, viewport_w - label_w - edge_margin_px))

            y = view.mapFromScene(QPointF(0.0, mid_y)).y()
            if last_bottom[side] is not None and y < last_bottom[side] + min_gap_px:
                y = last_bottom[side] + min_gap_px
            last_bottom[side] = y + label_h

            screen_pt = QPoint(int(x), int(y))
            txt_item.setPos(view.mapToScene(screen_pt))

    def _set_extended_line(self, line_item, p1, p2):
        start_p, end_p = extended_line_points(p1, p2, extend_len=250)
        if start_p is None:
            return
        line_item.setLine(start_p.x(), start_p.y(), end_p.x(), end_p.y())
