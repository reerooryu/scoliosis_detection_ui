# Annotated image export: the X-ray with the measurement overlay drawn on
# it, and a results panel on the right. Drawn straight from the result data,
# so it looks the same whatever the zoom or Edit Mode in the viewer. Landmark
# handles are not drawn.

import math
from datetime import datetime

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen, QPolygonF

from config import (
    COLOR_CORRIDOR_FILL, COLOR_CSVL_LINE, COLOR_VERTEBRA_OUTLINE, CURVE_COLORS,
    KP_BOTTOM_LEFT, KP_BOTTOM_RIGHT, KP_CENTER, KP_TOP_LEFT, KP_TOP_RIGHT,
)

# Sizes are for a 1000 px tall image; everything scales with the image height.
PANEL_WIDTH = 440
TEXT_DARK = QColor("#1b1f24")
TEXT_MUTED = QColor("#5f6b78")
RULE_COLOR = QColor("#d5dae0")


def render_annotated_image(pixmap, model_engine, curve_colors=None, source_name=None, edited=False):
    """The X-ray with its overlay, and the results panel on the right, as a
    QImage ready to save."""
    colors = list(curve_colors or CURVE_COLORS)
    img_w, img_h = pixmap.width(), pixmap.height()
    s = max(1.0, img_h / 1000.0)
    panel_w = int(round(PANEL_WIDTH * s))
    panel_h = _draw_panel(None, model_engine, colors, QRectF(0, 0, panel_w, 0), s, source_name, edited)
    out_h = max(img_h, int(math.ceil(panel_h)))

    out = QImage(img_w + panel_w, out_h, QImage.Format_RGB32)
    out.fill(QColor("white"))
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)
    painter.fillRect(QRectF(0, 0, img_w, out_h), QColor("black"))
    painter.drawPixmap(0, 0, pixmap)

    painter.save()
    painter.setClipRect(QRectF(0, 0, img_w, img_h))
    _draw_overlay(painter, model_engine, colors, img_w, img_h, s)
    painter.restore()

    _draw_panel(painter, model_engine, colors, QRectF(img_w, 0, panel_w, out_h), s, source_name, edited)
    painter.end()
    return out


# ---------------------------------------------------------------------------
# Overlay on the X-ray
# ---------------------------------------------------------------------------

def _font(size_px, bold=False):
    font = QFont()
    font.setPixelSize(max(1, int(round(size_px))))
    font.setBold(bold)
    return font


def _point(keypoint):
    return QPointF(keypoint[0], keypoint[1])


def _extended_line(p1, p2, extend):
    """The p1-p2 line, extended past both ends (like the viewer's Cobb lines)."""
    dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
    length = math.hypot(dx, dy)
    if length == 0:
        return None
    ux, uy = dx / length, dy / length
    return (QPointF(p1.x() - ux * extend, p1.y() - uy * extend),
            QPointF(p2.x() + ux * extend * 1.4, p2.y() + uy * extend * 1.4))


def _draw_overlay(painter, engine, colors, img_w, img_h, s):
    detections = engine.get_detections()
    complete = [d for d in detections if len(d.get("keypoints", [])) >= 5]

    # Spinal corridor (same as the viewer).
    if complete:
        corridor = QPolygonF()
        for d in complete:
            corridor.append(_point(d["keypoints"][KP_TOP_LEFT]))
            corridor.append(_point(d["keypoints"][KP_BOTTOM_LEFT]))
        for d in reversed(complete):
            corridor.append(_point(d["keypoints"][KP_BOTTOM_RIGHT]))
            corridor.append(_point(d["keypoints"][KP_TOP_RIGHT]))
        painter.setPen(QPen(QColor(0, 204, 150, 100), s))
        painter.setBrush(QColor(*COLOR_CORRIDOR_FILL))
        painter.drawPolygon(corridor)

    # Vertebra outlines. Their numbers are drawn later, on top of the lines.
    number_font = _font(12 * s, bold=True)
    number_tags = []
    for index, d in enumerate(detections):
        kp = d.get("keypoints", [])
        if len(kp) < 5:
            continue
        painter.setPen(QPen(QColor(*COLOR_VERTEBRA_OUTLINE), 1.5 * s))
        painter.setBrush(QColor(0, 120, 215, 15))
        painter.drawPolygon(QPolygonF([
            _point(kp[KP_TOP_LEFT]), _point(kp[KP_TOP_RIGHT]),
            _point(kp[KP_BOTTOM_RIGHT]), _point(kp[KP_BOTTOM_LEFT]),
        ]))
        left = min(kp[KP_TOP_LEFT][0], kp[KP_BOTTOM_LEFT][0])
        number_tags.append(("#{}".format(index), left - 6 * s, kp[KP_CENTER][1]))

    # CSVL.
    csvl_x = engine.get_csvl_x()
    if csvl_x is not None:
        painter.setPen(QPen(QColor(*COLOR_CSVL_LINE), 2 * s, Qt.DashDotLine))
        painter.drawLine(QPointF(csvl_x, 0), QPointF(csvl_x, img_h))

    # Cobb lines, then their labels on top.
    labels = []
    for idx, pair in enumerate(engine.get_angle_pairs()):
        u_idx = pair.get("upper_detection_index")
        l_idx = pair.get("lower_detection_index")
        if u_idx is None or l_idx is None:
            continue
        if not (0 <= u_idx < len(detections) and 0 <= l_idx < len(detections)):
            continue
        u_kp = detections[u_idx]["keypoints"]
        l_kp = detections[l_idx]["keypoints"]
        color = QColor(colors[idx % len(colors)])
        painter.setPen(QPen(color, 2.5 * s))
        for p1, p2 in ((u_kp[KP_TOP_LEFT], u_kp[KP_TOP_RIGHT]), (l_kp[KP_BOTTOM_LEFT], l_kp[KP_BOTTOM_RIGHT])):
            line = _extended_line(_point(p1), _point(p2), 0.13 * img_h)
            if line is not None:
                painter.drawLine(*line)

        lo, hi = sorted((u_idx, l_idx))
        xs = [kp[0] for d in detections[lo:hi + 1] for kp in d["keypoints"]]
        mid_y = (u_kp[KP_TOP_LEFT][1] + l_kp[KP_BOTTOM_LEFT][1]) / 2.0
        labels.append((idx, color, pair.get("cobb_angle", 0.0), min(xs), max(xs), mid_y))

    for text, right_x, center_y in number_tags:
        _draw_number_tag(painter, text, right_x, center_y, number_font, s)
    _draw_curve_labels(painter, labels, img_w, img_h, s)


def _draw_number_tag(painter, text, right_x, center_y, font, s):
    """Small white number on a dark box, right-aligned to right_x."""
    metrics = QFontMetricsF(font)
    pad = 3 * s
    w = metrics.horizontalAdvance(text) + 2 * pad
    h = metrics.height() + pad
    box = QRectF(right_x - w, center_y - h / 2, w, h)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(0, 0, 0, 150))
    painter.drawRoundedRect(box, 2 * s, 2 * s)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255, 230))
    painter.drawText(box, Qt.AlignCenter, text)


def _draw_curve_labels(painter, labels, img_w, img_h, s):
    """One label per curve beside its vertebrae: right side if there is room,
    else left. Labels on the same side are pushed down so they don't overlap."""
    title_font = _font(13 * s, bold=True)
    angle_font = _font(17 * s, bold=True)
    title_m, angle_m = QFontMetricsF(title_font), QFontMetricsF(angle_font)
    strip, pad, gap, margin = 8 * s, 6 * s, 16 * s, 6 * s
    last_bottom = {"left": None, "right": None}

    for idx, color, angle, min_x, max_x, mid_y in labels:
        title = "Curve {}".format(idx + 1)
        value = "{:.1f}°".format(angle)
        w = strip + 2 * pad + max(title_m.horizontalAdvance(title), angle_m.horizontalAdvance(value))
        h = 2 * pad + title_m.height() + angle_m.height()

        if max_x + gap + w <= img_w - margin:
            side, x = "right", max_x + gap
        else:
            side, x = "left", min_x - gap - w
        x = max(margin, min(x, img_w - w - margin))

        y = mid_y - h / 2
        if last_bottom[side] is not None and y < last_bottom[side] + 8 * s:
            y = last_bottom[side] + 8 * s
        y = max(margin, min(y, img_h - h - margin))
        last_bottom[side] = y + h

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 230))
        painter.drawRect(QRectF(x, y, w, h))
        painter.setBrush(color)
        painter.drawRect(QRectF(x, y, strip, h))
        painter.setPen(QColor("#333333"))
        painter.setFont(title_font)
        painter.drawText(QRectF(x + strip + pad, y + pad, w, title_m.height()), Qt.AlignLeft | Qt.AlignVCenter, title)
        painter.setPen(QColor("#111111"))
        painter.setFont(angle_font)
        painter.drawText(QRectF(x + strip + pad, y + pad + title_m.height(), w, angle_m.height()),
                         Qt.AlignLeft | Qt.AlignVCenter, value)


# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------

def _draw_panel(painter, engine, colors, rect, s, source_name, edited):
    """Draw the results panel in rect, or only measure it when painter is
    None. Returns the height it needs."""
    pad = 28 * s
    x = rect.left() + pad
    width = rect.width() - 2 * pad
    y = rect.top() + pad

    def text(value, size, bold=False, color=TEXT_DARK, after=0.35):
        nonlocal y
        font = _font(size * s, bold)
        box = QFontMetricsF(font).boundingRect(QRectF(0, 0, width, 100000), Qt.TextWordWrap, value)
        if painter is not None:
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(QRectF(x, y, width, box.height()), Qt.TextWordWrap, value)
        y += box.height() + after * size * s

    def rule():
        nonlocal y
        y += 6 * s
        if painter is not None:
            painter.setPen(QPen(RULE_COLOR, max(1.0, s)))
            painter.drawLine(QPointF(x, y), QPointF(x + width, y))
        y += 16 * s

    def row(label, value, chip=None):
        nonlocal y
        font, bold = _font(17 * s), _font(17 * s, bold=True)
        h = QFontMetricsF(font).height()
        label_x = x
        if chip is not None:
            label_x = x + h * 0.9
        if painter is not None:
            if chip is not None:
                painter.fillRect(QRectF(x, y + h * 0.2, h * 0.6, h * 0.6), QColor(chip))
            painter.setFont(font)
            painter.setPen(TEXT_DARK)
            painter.drawText(QRectF(label_x, y, width - (label_x - x), h), Qt.AlignLeft | Qt.AlignVCenter, label)
            painter.setFont(bold)
            painter.drawText(QRectF(x, y, width, h), Qt.AlignRight | Qt.AlignVCenter, value)
        y += h * 1.45

    pairs = engine.get_angle_pairs()
    apex_idx, deviation = engine.get_apex()

    text("Scoliosis Assessment", 26, bold=True)
    if source_name:
        text(source_name, 14, color=TEXT_MUTED, after=0.1)
    text("Exported " + datetime.now().strftime("%Y-%m-%d %H:%M"), 14, color=TEXT_MUTED)
    rule()

    text("Primary Cobb angle", 15, color=TEXT_MUTED, after=0.1)
    if pairs:
        text("{:.1f}°".format(engine.get_selected_cobb_angle()), 40, bold=True)
    else:
        text("No curve above 10°", 22, bold=True)
    rule()

    if pairs:
        for i, pair in enumerate(pairs):
            row("Curve {}  (#{}–#{})".format(i + 1, pair["upper_detection_index"], pair["lower_detection_index"]),
                "{:.1f}°".format(pair.get("cobb_angle", 0.0)), chip=colors[i % len(colors)])
        rule()

    row("Apex vertebra", "#{}".format(apex_idx) if apex_idx is not None else "-")
    row("CSVL deviation", "{:.1f} px".format(deviation) if apex_idx is not None else "-")
    row("Vertebrae detected", str(len(engine.get_detections())))
    rule()

    if edited:
        text("Landmarks were adjusted by hand after the AI result.", 14)
    text("AI-assisted measurement. Check before clinical use.", 13, color=TEXT_MUTED)
    text("Vertebrae are numbered from the top, starting at #0. CSVL deviation is in pixels.", 13, color=TEXT_MUTED)

    return y + pad - rect.top()
