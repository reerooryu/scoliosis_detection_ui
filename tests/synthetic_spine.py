# A made-up spine for the tests: no patient data. 23 vertebrae, with the
# marker vertebra (class 1) at #5 and an S-shaped curve whose Cobb angles are
# known in advance, because each vertebra's tilt is set directly.

import math

from modules.geometry import curve_start_index, select_angle_pairs

MARKER_INDEX = 5


def _tilts():
    """Endplate tilt of each vertebra, in degrees (upper = lower, no wedging)."""
    tilts = [10.0, -20.0, 5.0, -3.0, 2.0]            # above the marker: ignored
    tilts += [25.0 * i / 6 for i in range(7)]          # #5 to #11:   0 -> 25
    tilts += [25.0 - 65.0 * i / 7 for i in range(1, 8)]  # #12 to #18: down to -40
    tilts += [-40.0 + 10.0 * i for i in range(1, 5)]   # #19 to #22:  back up to 0
    return tilts


TILTS = _tilts()

# Built from TILTS: from #5 (0) up to #11 (25), down to #18 (-40), back to #22 (0).
EXPECTED_CURVES = [(5, 11, 25.0), (11, 18, 65.0), (18, 22, 40.0)]


def _corner(cx, cy, dx, dy, tilt):
    """Corner at offset (dx, dy) from the center, rotated by tilt. Image y
    grows downward, so a positive tilt lowers the right side."""
    a = math.radians(tilt)
    return [cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a), 1.0]


def make_result():
    """A result in the same layout the server returns."""
    width, height = 1200, 1800
    detections = []
    for i, tilt in enumerate(TILTS):
        cx = 600 + 120 * math.sin(i / 3.0)
        cy = 100 + i * 70
        w, h = 60 + i * 2, 40 + i
        corners = [
            _corner(cx, cy, -w / 2, -h / 2, tilt),   # top-left
            _corner(cx, cy, w / 2, -h / 2, tilt),    # top-right
            _corner(cx, cy, -w / 2, h / 2, tilt),    # bottom-left
            _corner(cx, cy, w / 2, h / 2, tilt),     # bottom-right
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        detections.append({
            "index": i,
            "class": 1 if i == MARKER_INDEX else 0,
            "score": 0.99,
            "box": [min(xs), min(ys), max(xs), max(ys)],
            "upper_oblique": tilt,
            "lower_oblique": tilt,
            "keypoints": [[cx, cy, 1.0]] + corners,
        })

    classes = [d["class"] for d in detections]
    pairs = select_angle_pairs(TILTS, TILTS, curve_start_index(classes))
    return {
        "input_shape": [height, width],
        "detection_count": len(detections),
        "detections": detections,
        "angle_pairs": pairs,
        "all_angles": [p["cobb_angle"] for p in pairs],
        "selected_cobb_angle": max(p["cobb_angle"] for p in pairs),
        "upper_obliques": list(TILTS),
        "lower_obliques": list(TILTS),
    }
