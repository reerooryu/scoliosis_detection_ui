# Drawing rules shared by the viewer (modules/overlay.py) and the exported
# JPG (modules/report_image.py), so both show the same thing.

import math

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor

DARK_INK = QColor("#111111")
LIGHT_INK = QColor("white")
MAIN_APEX_FILL = QColor(255, 255, 255, 235)   # whole-spine apex outside any curve


def curve_color(colors, curve_index):
    """Color of a curve. With more curves than colors, colors repeat."""
    return colors[curve_index % len(colors)]


def ink_for(fill):
    """Text color that stays readable on fill: dark on light, light on dark."""
    return DARK_INK if fill.lightness() > 140 else LIGHT_INK


def apex_tag_style(role, colors, fill_alpha=255):
    """(fill, ink, large) for a vertebra's level tag, from its apex role
    (ScoliosisModelEngine.get_apex_roles()). A curve's apex is filled with
    the curve's color; the whole-spine apex is drawn larger. None for an
    ordinary vertebra, which keeps the faint default style."""
    if role is None:
        return None
    curve_index, is_main_apex = role
    if curve_index is None:
        return QColor(MAIN_APEX_FILL), DARK_INK, True
    fill = QColor(curve_color(colors, curve_index))
    fill.setAlpha(fill_alpha)
    return fill, ink_for(fill), is_main_apex


def extended_line(p1, p2, before, after):
    """The line through QPointF p1 and p2, extended by `before` past p1 and
    `after` past p2, as (start, end). None if the two points are the same."""
    dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
    length = math.hypot(dx, dy)
    if length == 0:
        return None
    ux, uy = dx / length, dy / length
    return (QPointF(p1.x() - ux * before, p1.y() - uy * before),
            QPointF(p2.x() + ux * after, p2.y() + uy * after))
