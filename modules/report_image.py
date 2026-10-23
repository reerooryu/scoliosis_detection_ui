# Annotated image export: the X-ray with the measurement overlay drawn on
# it, and a narrow dark results strip on the right. The X-ray is the focus;
# the strip holds only the key numbers. Drawn straight from the result data,
# so it looks the same whatever the zoom or Edit Mode in the viewer.

import math
from datetime import datetime

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen, QPolygonF

from config import (
    COLOR_CSVL_LINE, COLOR_VERTEBRA_OUTLINE, CURVE_COLORS,
    KP_BOTTOM_LEFT, KP_BOTTOM_RIGHT, KP_CENTER, KP_TOP_LEFT, KP_TOP_RIGHT,
)

# Sizes are for a 1000 px tall image; everything scales with the image height.
PANEL_BG = QColor("#0d0f12")
TEXT_MAIN = QColor("#f2f4f7")
TEXT_MUTED = QColor("#8a94a0")
TEXT_EDITED = QColor("#ffb74d")


def render_annotated_image(pixmap, model_engine, curve_colors=None, source_name=None, edited=False):
    """The X-ray with its overlay, and a narrow results strip on the right,
    as a QImage ready to save."""
    colors = list(curve_colors or CURVE_COLORS)
    img_w, img_h = pixmap.width(), pixmap.height()
    s = max(1.0, img_h / 1000.0)
    panel = _Panel(model_engine, colors, s, source_name, edited)
    panel_w = int(math.ceil(panel.width))
    out_h = max(img_h, int(math.ceil(panel.height)))

    out = QImage(img_w + panel_w, out_h, QImage.Format_RGB32)
    out.fill(PANEL_BG)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)
    painter.fillRect(QRectF(0, 0, img_w, out_h), QColor("black"))
    painter.drawPixmap(0, 0, pixmap)

    painter.save()
    painter.setClipRect(QRectF(0, 0, img_w, img_h))
    _draw_overlay(painter, model_engine, colors, img_w, img_h, s)
    painter.restore()

    panel.draw(painter, QRectF(img_w, 0, panel_w, out_h))
    painter.end()
    return out


def _font(size_px, bold=False):
    font = QFont()
    font.setPixelSize(max(1, int(round(size_px))))
    font.setBold(bold)
    return font


def _point(keypoint):
    return QPointF(keypoint[0], keypoint[1])


# ---------------------------------------------------------------------------
# Overlay on the X-ray (kept light so the X-ray stays readable)
# ---------------------------------------------------------------------------

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

    # Vertebra outlines. Their level names are drawn later, on top of the lines.
    number_tags = []
    painter.setPen(QPen(QColor(*COLOR_VERTEBRA_OUTLINE), 1.0 * s))
    painter.setBrush(QColor(0, 120, 215, 12))
    for index, d in enumerate(detections):
        kp = d.get("keypoints", [])
        if len(kp) < 5:
            continue
        painter.drawPolygon(QPolygonF([
            _point(kp[KP_TOP_LEFT]), _point(kp[KP_TOP_RIGHT]),
            _point(kp[KP_BOTTOM_RIGHT]), _point(kp[KP_BOTTOM_LEFT]),
        ]))
        left = min(kp[KP_TOP_LEFT][0], kp[KP_BOTTOM_LEFT][0])
        number_tags.append((index, (engine.get_level(index), left - 5 * s, kp[KP_CENTER][1])))

    # CSVL.
    csvl_x = engine.get_csvl_x()
    if csvl_x is not None:
        painter.setPen(QPen(QColor(*COLOR_CSVL_LINE), 1.5 * s, Qt.DashDotLine))
        painter.drawLine(QPointF(csvl_x, 0), QPointF(csvl_x, img_h))

    # Cobb lines.
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
        painter.setPen(QPen(color, 2.0 * s))
        for p1, p2 in ((u_kp[KP_TOP_LEFT], u_kp[KP_TOP_RIGHT]), (l_kp[KP_BOTTOM_LEFT], l_kp[KP_BOTTOM_RIGHT])):
            line = _extended_line(_point(p1), _point(p2), 0.13 * img_h)
            if line is not None:
                painter.drawLine(*line)

        lo, hi = sorted((u_idx, l_idx))
        xs = [kp[0] for d in detections[lo:hi + 1] for kp in d["keypoints"]]
        mid_y = (u_kp[KP_TOP_LEFT][1] + l_kp[KP_BOTTOM_LEFT][1]) / 2.0
        labels.append((color, pair.get("cobb_angle", 0.0), min(xs), max(xs), mid_y))

    # Level names. Each curve's apex is filled with the curve's color; the
    # whole-spine apex (furthest from the CSVL) is also drawn larger.
    main_apex, _ = engine.get_apex()
    apex_colors = {
        apex_idx: QColor(colors[idx % len(colors)])
        for idx, apex_idx in enumerate(engine.get_curve_apexes())
        if apex_idx is not None
    }

    number_font = _font(9 * s, bold=True)
    main_apex_font = _font(11.5 * s, bold=True)
    for index, (text, right_x, center_y) in number_tags:
        font = main_apex_font if index == main_apex else number_font
        if index in apex_colors:
            fill = QColor(apex_colors[index])
            fill.setAlpha(235)
            ink = QColor("#111111") if fill.lightness() > 140 else QColor("white")
            _draw_number_tag(painter, text, right_x, center_y, font, s, fill, ink)
        elif index == main_apex:
            _draw_number_tag(painter, text, right_x, center_y, font, s, QColor(255, 255, 255, 235), QColor("#111111"))
        else:
            _draw_number_tag(painter, text, right_x, center_y, font, s)
    _draw_curve_labels(painter, labels, img_w, img_h, s)


def _draw_number_tag(painter, text, right_x, center_y, font, s,
                     fill=QColor(0, 0, 0, 110), ink=QColor(255, 255, 255, 200)):
    """Small vertebra level name on a box, right-aligned to right_x. Faint
    by default; apex vertebrae pass their own colors."""
    metrics = QFontMetricsF(font)
    pad = 2 * s
    w = metrics.horizontalAdvance(text) + 2 * pad
    h = metrics.height()
    box = QRectF(right_x - w, center_y - h / 2, w, h)
    painter.setPen(Qt.NoPen)
    painter.setBrush(fill)
    painter.drawRoundedRect(box, 2 * s, 2 * s)
    painter.setFont(font)
    painter.setPen(ink)
    painter.drawText(box, Qt.AlignCenter, text)


def _draw_curve_labels(painter, labels, img_w, img_h, s):
    """A small angle tag beside each curve, in the curve's color: right side
    if there is room, else left. Tags on the same side don't overlap."""
    font = _font(13 * s, bold=True)
    metrics = QFontMetricsF(font)
    strip, pad, gap, margin = 4 * s, 4 * s, 12 * s, 6 * s
    last_bottom = {"left": None, "right": None}

    for color, angle, min_x, max_x, mid_y in labels:
        value = "{:.1f}°".format(angle)
        w = strip + 2 * pad + metrics.horizontalAdvance(value)
        h = metrics.height() + pad

        if max_x + gap + w <= img_w - margin:
            side, x = "right", max_x + gap
        else:
            side, x = "left", min_x - gap - w
        x = max(margin, min(x, img_w - w - margin))

        y = mid_y - h / 2
        if last_bottom[side] is not None and y < last_bottom[side] + 6 * s:
            y = last_bottom[side] + 6 * s
        y = max(margin, min(y, img_h - h - margin))
        last_bottom[side] = y + h

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRect(QRectF(x, y, w, h))
        painter.setBrush(color)
        painter.drawRect(QRectF(x, y, strip, h))
        painter.setFont(font)
        painter.setPen(TEXT_MAIN)
        painter.drawText(QRectF(x + strip + pad, y, w - strip - pad, h), Qt.AlignLeft | Qt.AlignVCenter, value)


# ---------------------------------------------------------------------------
# Results strip
# ---------------------------------------------------------------------------

class _Panel:
    """The narrow results strip: key numbers at the top, a small footer at
    the bottom. Its width is just what the content needs."""

    def __init__(self, engine, colors, s, source_name, edited):
        self.s = s
        self.pad = 16 * s
        label, big, angle = _font(11 * s), _font(30 * s, bold=True), _font(15 * s, bold=True)
        small, small_bold, foot = _font(11 * s), _font(11 * s, bold=True), _font(9.5 * s)

        pairs = engine.get_angle_pairs()
        apex_idx, deviation = engine.get_apex()

        # Each row is a list of segments: ("text", text, font, color) or
        # ("chip", color, font). None is a small gap.
        self.top = []
        if pairs:
            self.top.append([("text", "Cobb", label, TEXT_MUTED)])
            self.top.append([("text", "{:.1f}°".format(engine.get_selected_cobb_angle()), big, TEXT_MAIN)])
            self.top.append(None)
            for i, (pair, curve_apex_idx) in enumerate(zip(pairs, engine.get_curve_apexes())):
                row = [
                    ("chip", colors[i % len(colors)], angle),
                    ("text", "{:.1f}°".format(pair.get("cobb_angle", 0.0)), angle, TEXT_MAIN),
                    ("text", "  {}–{}".format(engine.get_level(pair["upper_detection_index"]),
                                               engine.get_level(pair["lower_detection_index"])), small, TEXT_MUTED),
                ]
                if curve_apex_idx is not None:
                    row.append(("text", "  apex ", small, TEXT_MUTED))
                    if curve_apex_idx == apex_idx:
                        row.append(("text", engine.get_level(curve_apex_idx), small_bold, TEXT_MAIN))
                    else:
                        row.append(("text", engine.get_level(curve_apex_idx), small, TEXT_MUTED))
                self.top.append(row)
        else:
            self.top.append([("text", "No curve above 10°", angle, TEXT_MAIN)])
        self.top.append(None)
        if apex_idx is not None:
            self.top.append([("text", "Apex ", small, TEXT_MUTED),
                             ("text", engine.get_level(apex_idx), small_bold, TEXT_MAIN)])
            self.top.append([("text", "CSVL {:.0f} px".format(deviation), small, TEXT_MUTED)])
        seconds = engine.get_processing_time()
        if seconds is not None:
            self.top.append([("text", "Processing {:.2f} s".format(seconds), small, TEXT_MUTED)])
        if edited:
            self.top.append([("text", "Edited by hand", small, TEXT_EDITED)])

        self.source_name = source_name
        self.foot_font = foot
        self.bottom = [
            [("text", datetime.now().strftime("%Y-%m-%d %H:%M"), foot, TEXT_MUTED)],
            [("text", "Levels counted up from L5", foot, TEXT_MUTED)],
            [("text", "AI-assisted · verify", foot, TEXT_MUTED)],
        ]

        content_w = max(self._row_width(row) for row in self.top + self.bottom if row)
        self.width = content_w + 2 * self.pad
        name_h = QFontMetricsF(foot).height() * 1.25 if source_name else 0
        self.height = (2 * self.pad + self._rows_height(self.top) + 2 * self.pad
                       + name_h + self._rows_height(self.bottom))

    def _gap(self):
        return 8 * self.s

    @staticmethod
    def _row_height(row):
        return max(QFontMetricsF(seg[2]).height() for seg in row) * 1.25

    def _rows_height(self, rows):
        return sum(self._gap() if row is None else self._row_height(row) for row in rows)

    def _chip_size(self, font):
        return QFontMetricsF(font).height() * 0.55

    def _row_width(self, row):
        width = 0.0
        for seg in row:
            if seg[0] == "chip":
                width += self._chip_size(seg[2]) + 6 * self.s
            else:
                width += QFontMetricsF(seg[2]).horizontalAdvance(seg[1])
        return width

    def _draw_rows(self, painter, rows, x, y):
        for row in rows:
            if row is None:
                y += self._gap()
                continue
            h = self._row_height(row)
            cx = x
            for seg in row:
                if seg[0] == "chip":
                    size = self._chip_size(seg[2])
                    painter.fillRect(QRectF(cx, y + (h - size) / 2, size, size), QColor(seg[1]))
                    cx += size + 6 * self.s
                else:
                    _, text, font, color = seg
                    seg_w = QFontMetricsF(font).horizontalAdvance(text)
                    painter.setFont(font)
                    painter.setPen(color)
                    painter.drawText(QRectF(cx, y, seg_w + 2, h), Qt.AlignLeft | Qt.AlignBottom, text)
                    cx += seg_w
            y += h
        return y

    def draw(self, painter, rect):
        x = rect.left() + self.pad
        self._draw_rows(painter, self.top, x, rect.top() + self.pad)

        # Footer at the bottom: file name (shortened to fit), date, note.
        bottom_h = self._rows_height(self.bottom)
        metrics = QFontMetricsF(self.foot_font)
        name_h = metrics.height() * 1.25 if self.source_name else 0
        y = rect.bottom() - self.pad - bottom_h - name_h
        if self.source_name:
            name = metrics.elidedText(self.source_name, Qt.ElideMiddle, rect.width() - 2 * self.pad)
            painter.setFont(self.foot_font)
            painter.setPen(TEXT_MUTED)
            painter.drawText(QRectF(x, y, rect.width() - 2 * self.pad, name_h), Qt.AlignLeft | Qt.AlignBottom, name)
            y += name_h
        self._draw_rows(painter, self.bottom, x, y)
