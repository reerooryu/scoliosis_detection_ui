# Measurement overlay graphics items + rendering orchestration.
#
# These QGraphicsItem subclasses render AI-predicted landmarks and Cobb
# measurement lines on top of the base image from modules/canvas.py. They
# are drawn as their own scene items and never modify the underlying X-ray
# pixmap. `OverlayLayer` at the bottom of this file owns and updates those
# items for a given ImageCanvas + ScoliosisModelEngine pair, and is what
# modules/main_window.py drives after a live/mock inference result comes in.

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
    """Emitted when a landmark is dragged in Edit mode: (det_idx, kp_idx, x, y)."""
    keypoint_moved = Signal(int, int, float, float)
    # Emitted once per drag gesture (mouse press / release on a handle), not
    # per pixel of movement -- used to snapshot undo state and to trigger a
    # full, safe overlay rebuild once the drag is no longer active.
    drag_started = Signal(int, int)
    drag_finished = Signal(int, int)


class LandmarkHandleItem(QGraphicsEllipseItem):
    """Circle handle representing a single vertebra keypoint.

    The 4 corner keypoints are draggable. The center keypoint (KP_CENTER) is
    not: ScoliosisModelEngine.recalculate_all_metrics() always recomputes it
    as the average of the 4 corner keypoints, so it is a derived value, not a
    source of truth. It's still rendered -- as a visual reference showing
    where that average currently sits -- but is non-interactive. (It used to
    be draggable like any other handle; the drag appeared to work live but
    was silently discarded on the very next recalculation, while still
    leaving behind a spurious undo snapshot and a "dirty" flag.)
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

        # Keeps the handle's on-screen size constant regardless of the
        # canvas's zoom level. Without this, the ellipse's radius is in
        # scene coordinates like everything else, so it grows right
        # along with the image -- at a few clicks of zoom-in (exactly
        # when you'd want Edit Mode for precision) the handles quickly
        # became huge. HANDLE_RADIUS/ACTIVE_HANDLE_RADIUS are screen
        # pixels now, not scene units.
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
    """Returns (start, end) QPointF pair extending the p1->p2 line outward,
    for drawing a full Cobb-angle intersecting plane."""
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
    """Owns and renders the landmark/Cobb-angle graphics items for one
    ImageCanvas, driven by a ScoliosisModelEngine's detection data.

    Kept separate from ImageCanvas (dumb base image viewer) and the model
    engine (data/math) so each stays single-purpose: this class is purely
    "given these detections, draw/update these scene items."
    """

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
        self.interactive_mode = False

    def clear(self):
        """Remove tracked overlay items without touching deleted C++ wrappers.

        ``QGraphicsScene.clear()`` deletes scene-owned QGraphicsItems.  The
        corresponding Python wrappers can briefly remain in our lists, so
        calling a Qt method on one would raise ``RuntimeError: Internal C++
        object already deleted``.  Test the wrapper before interacting with
        it; clearing our Python references is still required in either case.
        """
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

    @staticmethod
    def _remove_item_if_valid(scene, item):
        """Detach ``item`` only while its underlying C++ instance exists."""
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
        """Creates the overlay items on first call, updates positions on
        subsequent calls (e.g. after a drag-triggered recalculation)."""
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
                    # Skip the handle currently being dragged: Qt is already
                    # moving it as part of the user's live mouse drag, so its
                    # position here is already correct (this update loop is
                    # itself triggered by that very handle's itemChange()).
                    # Calling setPos() on it again re-enters itemChange() for
                    # the same item while Qt's own mouse-grab/drag handling
                    # for it is still on the call stack -- exactly the kind
                    # of scene mutation reentrancy that Qt Graphics View
                    # doesn't handle safely (this was the "adjusting points
                    # freezes/crashes the app" bug). No blockSignals() needed
                    # either way: QGraphicsEllipseItem isn't a QObject.
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
        """Updates the Cobb measurement lines/labels. Reuses existing scene
        items in place (setLine/setHtml) rather than destroying and
        recreating them on every call. The pairing/count of curves never
        changes from local keypoint edits (that's fixed by the server's
        curve-detection pass), so a full destroy+recreate here was pure
        waste on every single mouse-move tick of a drag -- and, worse, meant
        mutating the scene's item list synchronously from inside another
        item's itemChange() handler on every one of those ticks, which is
        exactly the kind of reentrant scene mutation Qt Graphics View
        doesn't handle safely (see the comment on the handle-skip in
        render() -- same root cause, different symptom)."""
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
                # Keeps the label a constant, readable size on screen
                # regardless of canvas zoom -- same fix as the landmark
                # handles. Without this, a label's HTML font-size is in
                # scene units like everything else, so on a large image
                # that fits the view at a small scale, the label shrinks
                # right along with it (reported as "abnormally small").
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

            # The vertebrae spanned by *this* curve (not the whole spine --
            # a lower/upper curve in a real S-shaped scoliosis case can
            # bulge in opposite directions) define how far out the label
            # needs to sit to clear the anatomy at that height.
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

        self._place_labels_without_overlap(pending_labels)

    def _place_labels_without_overlap(self, pending_labels, min_gap_px=10, hug_gap_px=16, edge_margin_px=6):
        """Places each Cobb-angle label right next to its own curve, hugging
        whichever side (left or right of that curve's vertebrae) has more
        clear space in the viewport -- instead of either (a) a fixed scene-
        unit offset from the spine, which only cleared the ribcage on the
        narrow bundled demo image and landed right on top of the anatomy on
        a wider clinical X-ray, or (b) a global left-margin anchor, which
        keeps labels clear of anatomy but detaches them from the curve
        they're describing.

        For each curve, `local_min_x`/`local_max_x` are the scene-space
        horizontal extent of just the vertebrae spanned by that curve (not
        the whole spine -- an S-shaped curve's upper and lower segments can
        bulge in opposite directions). Whichever side of that extent has
        more room out to the viewport edge gets the label, hugging the
        vertebra edge by a small fixed screen-pixel gap so it reads as
        "attached to this curve" rather than floating. Labels are then
        nudged down (per side) if they'd collide with the previous label
        placed on that same side, and finally clamped to stay fully inside
        the viewport as a fallback for extreme cases (e.g. a curve that
        fills almost the entire image width).
        """
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

            # Fallback: if neither side actually had room (a curve spanning
            # almost the full image width), keep the label on-screen rather
            # than letting it run off the viewport edge.
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
