# Custom Interactive Canvas and Graphics Items using Qt Graphics View Framework

from PySide6.QtCore import Qt, QPointF, Signal, QObject, QTimer
from PySide6.QtGui import QPen, QBrush, QColor, QPolygonF, QPainterPath, QTransform, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPolygonItem, QGraphicsPathItem, QGraphicsLineItem, QGraphicsTextItem, QGraphicsPixmapItem
import math
from config import (
    KP_CENTER, KP_TOP_LEFT, KP_TOP_RIGHT, KP_BOTTOM_LEFT, KP_BOTTOM_RIGHT,
    COLOR_VERTEBRA_OUTLINE, COLOR_CORRIDOR_FILL, COLOR_KEYPOINT_CENTER,
    COLOR_KEYPOINT_CORNER, COLOR_COBB_LINE, COLOR_COBB_TEXT_BG,
    HANDLE_RADIUS, ACTIVE_HANDLE_RADIUS
)

class CanvasSignals(QObject):
    # Emitted when a keypoint is dragged. Arguments: (detection_index, keypoint_index, new_x, new_y)
    keypoint_moved = Signal(int, int, float, float)
    # Emitted when zoom factor changes
    zoom_changed = Signal(float)


class VertebraKeypointItem(QGraphicsEllipseItem):
    """Draggable circle handle representing a single keypoint on a vertebra."""
    def __init__(self, det_idx, kp_idx, x, y, parent_item=None, signals=None):
        super().__init__(-HANDLE_RADIUS, -HANDLE_RADIUS, HANDLE_RADIUS * 2, HANDLE_RADIUS * 2, parent_item)
        self.det_idx = det_idx
        self.kp_idx = kp_idx
        self.signals = signals
        self.is_dragging = False

        # Visual appearance
        self.setPen(QPen(QColor(0, 0, 0, 180), 1))
        
        # Color code based on role (Center vs Corner)
        if kp_idx == KP_CENTER:
            self.setBrush(QBrush(QColor(*COLOR_KEYPOINT_CENTER)))
        else:
            self.setBrush(QBrush(QColor(*COLOR_KEYPOINT_CORNER)))

        # Interaction configuration
        self.setFlags(
            QGraphicsEllipseItem.ItemIsMovable |
            QGraphicsEllipseItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setPos(x, y)

    def hoverEnterEvent(self, event):
        """Grow handle size slightly on hover."""
        self.setRect(-ACTIVE_HANDLE_RADIUS, -ACTIVE_HANDLE_RADIUS, ACTIVE_HANDLE_RADIUS * 2, ACTIVE_HANDLE_RADIUS * 2)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Restore handle size on hover leave."""
        if not self.is_dragging:
            self.setRect(-HANDLE_RADIUS, -HANDLE_RADIUS, HANDLE_RADIUS * 2, HANDLE_RADIUS * 2)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self.is_dragging = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.setRect(-HANDLE_RADIUS, -HANDLE_RADIUS, HANDLE_RADIUS * 2, HANDLE_RADIUS * 2)
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        """Catches movement and emits dynamic signal to update model & scene."""
        if change == QGraphicsEllipseItem.ItemPositionChange and self.is_dragging:
            new_pos = value
            if self.signals:
                self.signals.keypoint_moved.emit(self.det_idx, self.kp_idx, new_pos.x(), new_pos.y())
        return super().itemChange(change, value)


class VertebraOutlineItem(QGraphicsPolygonItem):
    """Draws a polygon outlining the 4 corners of a single vertebra."""
    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self.setPen(QPen(QColor(*COLOR_VERTEBRA_OUTLINE), 1.5))
        self.setBrush(QBrush(QColor(0, 120, 215, 15))) # Very subtle translucent fill

    def update_polygon(self, keypoints):
        """Connects corners: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left -> Top-Left."""
        if len(keypoints) >= 5:
            poly = QPolygonF()
            poly.append(QPointF(keypoints[KP_TOP_LEFT][0], keypoints[KP_TOP_LEFT][1]))
            poly.append(QPointF(keypoints[KP_TOP_RIGHT][0], keypoints[KP_TOP_RIGHT][1]))
            poly.append(QPointF(keypoints[KP_BOTTOM_RIGHT][0], keypoints[KP_BOTTOM_RIGHT][1]))
            poly.append(QPointF(keypoints[KP_BOTTOM_LEFT][0], keypoints[KP_BOTTOM_LEFT][1]))
            self.setPolygon(poly)


class SpinalCorridorItem(QGraphicsPolygonItem):
    """A continuous semi-transparent overlay covering the entire spinal canal corridor."""
    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self.setPen(QPen(QColor(0, 204, 150, 100), 1))
        self.setBrush(QBrush(QColor(*COLOR_CORRIDOR_FILL)))

    def update_corridor(self, detections):
        """
        Creates a closed corridor path surrounding the vertebrae.
        Combines top-left to bottom-left on left side, and bottom-right to top-right on right side.
        """
        if not detections:
            return
            
        poly = QPolygonF()
        
        # 1. Left Boundary (Top-Left and Bottom-Left of each vertebra from top to bottom)
        for det in detections:
            kps = det.get("keypoints", [])
            if len(kps) >= 5:
                poly.append(QPointF(kps[KP_TOP_LEFT][0], kps[KP_TOP_LEFT][1]))
                poly.append(QPointF(kps[KP_BOTTOM_LEFT][0], kps[KP_BOTTOM_LEFT][1]))
                
        # 2. Right Boundary (Bottom-Right and Top-Right of each vertebra from bottom to top)
        for det in reversed(detections):
            kps = det.get("keypoints", [])
            if len(kps) >= 5:
                poly.append(QPointF(kps[KP_BOTTOM_RIGHT][0], kps[KP_BOTTOM_RIGHT][1]))
                poly.append(QPointF(kps[KP_TOP_RIGHT][0], kps[KP_TOP_RIGHT][1]))
                
        self.setPolygon(poly)


class CobbMeasurementLineItem(QGraphicsLineItem):
    """Draws a measurement line for a Cobb angle from a vertebra's endplates."""
    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self.setPen(QPen(QColor(*COLOR_COBB_LINE), 2, Qt.DashLine))


class ScoliosisInteractiveCanvas(QGraphicsView):
    """High-performance graphics canvas supporting drag adjustment, zoom, and medical overlays."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.signals = CanvasSignals()
        
        # UI controls
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        
        # Layer items
        self.image_item = None
        self.corridor_item = None
        self.outline_items = []
        self.handle_items = []
        self.cobb_lines = []
        self.cobb_texts = []
        
        # Interactive Mode flag (True enables keypoint dragging handles)
        self.interactive_mode = False
        
        # Zoom parameters
        self.zoom_factor = 1.0

    def set_interactive(self, enabled):
        """Enables or disables keypoint drag handles (Page 2 vs Page 3)."""
        self.interactive_mode = enabled
        for handle in self.handle_items:
            handle.setVisible(enabled)

    def load_image(self, pixmap):
        """Sets the base original image layer."""
        self.scene.clear()
        
        # Re-initialize item lists
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        
        # Set scene bounding rect to match image dimensions
        self.scene.setSceneRect(self.image_item.boundingRect())
        
        # Create persistent overlay layers (Z-indexed)
        self.corridor_item = SpinalCorridorItem()
        self.corridor_item.setZValue(1) # Under vertebrae and handles
        self.scene.addItem(self.corridor_item)
        
        self.outline_items = []
        self.handle_items = []
        self.cobb_lines = []
        self.cobb_texts = []
        
        self.fit_in_view()

    def render_model_data(self, model_engine):
        """Populates the canvas with vertebrae outlines, drag handles, and measurement overlays."""
        detections = model_engine.get_detections()
        angle_pairs = model_engine.get_angle_pairs()
        
        # 1. Update/Create Spinal Corridor
        self.corridor_item.update_corridor(detections)
        
        # 2. Update/Create Vertebra outlines and handles
        if not self.outline_items:
            # First-time initialization
            for d_idx, det in enumerate(detections):
                outline = VertebraOutlineItem()
                outline.setZValue(2) # Over corridor
                outline.update_polygon(det["keypoints"])
                self.scene.addItem(outline)
                self.outline_items.append(outline)
                
                # Add drag handles for 5 keypoints
                for kp_idx, kp in enumerate(det["keypoints"]):
                    handle = VertebraKeypointItem(
                        d_idx, kp_idx, kp[0], kp[1], 
                        signals=self.signals
                    )
                    handle.setZValue(3) # Topmost layer
                    handle.setVisible(self.interactive_mode)
                    self.scene.addItem(handle)
                    self.handle_items.append(handle)
        else:
            # Real-time update (no item recreation to optimize speed)
            for idx, det in enumerate(detections):
                self.outline_items[idx].update_polygon(det["keypoints"])
                
            # Update handle positions
            handle_idx = 0
            for det in detections:
                for kp in det["keypoints"]:
                    # Temporarily block signals during programmatic coordinate updates
                    self.handle_items[handle_idx].blockSignals(True)
                    self.handle_items[handle_idx].setPos(kp[0], kp[1])
                    self.handle_items[handle_idx].blockSignals(False)
                    handle_idx += 1

        # 3. Update/Create Cobb Angle Measurement Annotations
        self.render_cobb_angle_overlays(detections, angle_pairs)

    def render_cobb_angle_overlays(self, detections, angle_pairs):
        """Draws endplate extension lines and angle clinical labels."""
        # Clear previous cobb lines
        for item in self.cobb_lines:
            self.scene.removeItem(item)
        for item in self.cobb_texts:
            self.scene.removeItem(item)
        self.cobb_lines = []
        self.cobb_texts = []
        
        for idx, pair in enumerate(angle_pairs):
            u_idx = pair.get("upper_detection_index")
            l_idx = pair.get("lower_detection_index")
            cobb_val = pair.get("cobb_angle", 0.0)
            
            if 0 <= u_idx < len(detections) and 0 <= l_idx < len(detections):
                u_det = detections[u_idx]
                l_det = detections[l_idx]
                
                u_kps = u_det["keypoints"]
                l_kps = l_det["keypoints"]
                
                # Draw Line 1 along upper endplate (Keypoint 1 to Keypoint 2)
                p1_u = QPointF(u_kps[KP_TOP_LEFT][0], u_kps[KP_TOP_LEFT][1])
                p2_u = QPointF(u_kps[KP_TOP_RIGHT][0], u_kps[KP_TOP_RIGHT][1])
                line_u = self.draw_extended_line(p1_u, p2_u, extend_len=250)
                self.cobb_lines.append(line_u)
                
                # Draw Line 2 along lower endplate (Keypoint 3 to Keypoint 4)
                p1_l = QPointF(l_kps[KP_BOTTOM_LEFT][0], l_kps[KP_BOTTOM_LEFT][1])
                p2_l = QPointF(l_kps[KP_BOTTOM_RIGHT][0], l_kps[KP_BOTTOM_RIGHT][1])
                line_l = self.draw_extended_line(p1_l, p2_l, extend_len=250)
                self.cobb_lines.append(line_l)
                
                # Position Cobb Angle clinical text box near the middle of the curve
                mid_x = (p1_u.x() + p1_l.x()) / 2.0 - 150 # Shift left for readability
                mid_y = (p1_u.y() + p1_l.y()) / 2.0
                
                txt_item = QGraphicsTextItem()
                txt_item.setHtml(
                    f"<div style='background-color: rgba(255, 255, 255, 210); "
                    f"border: 1.5px solid rgb(255, 87, 34); padding: 4px; border-radius: 4px;'>"
                    f"<b style='color: rgb(255, 87, 34); font-size: 11px;'>Cobb Angle Pair #{idx+1}</b><br/>"
                    f"<span style='color: #333; font-size: 13px; font-weight: bold;'>Angle: {cobb_val:.2f}°</span>"
                    f"</div>"
                )
                txt_item.setPos(mid_x, mid_y)
                txt_item.setZValue(4) # Draw on top
                self.scene.addItem(txt_item)
                self.cobb_texts.append(txt_item)

    def draw_extended_line(self, p1, p2, extend_len=200):
        """Draws a line extended outwards to clearly visualize the Cobb Angle intersecting plane."""
        # Calculate line vector
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return None
            
        ux = dx / length
        uy = dy / length
        
        # Extend in both directions
        start_p = QPointF(p1.x() - ux * extend_len, p1.y() - uy * extend_len)
        end_p = QPointF(p2.x() + ux * (extend_len + 100), p2.y() + uy * (extend_len + 100))
        
        g_line = CobbMeasurementLineItem()
        g_line.setLine(start_p.x(), start_p.y(), end_p.x(), end_p.y())
        g_line.setZValue(2.5) # Above outlines
        self.scene.addItem(g_line)
        return g_line

    def fit_in_view(self):
        """Fits the entire graphics image properly centered inside the widget viewport."""
        if self.image_item:
            self.fitInView(self.image_item, Qt.KeepAspectRatio)
            self.zoom_factor = self.transform().m11()
            self.signals.zoom_changed.emit(self.zoom_factor)

    def zoom_in(self):
        self.scale(1.15, 1.15)
        self.zoom_factor = self.transform().m11()
        self.signals.zoom_changed.emit(self.zoom_factor)

    def zoom_out(self):
        self.scale(1.0 / 1.15, 1.0 / 1.15)
        self.zoom_factor = self.transform().m11()
        self.signals.zoom_changed.emit(self.zoom_factor)

    def wheelEvent(self, event):
        """Standard CTRL+Scroll zoom gesture."""
        if event.modifiers() == Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event):
        """Maintains aspect ratio during viewport sizing."""
        super().resizeEvent(event)
        self.fit_in_view()

    def showEvent(self, event):
        """Ensure the image fits the view perfectly as soon as the canvas is displayed."""
        super().showEvent(event)
        self.fit_in_view()
        # Ensure it fits even after the parent layouts / splitters settle completely
        QTimer.singleShot(100, self.fit_in_view)
