# Clinical geometry calculations that sit apart from I/O (modules/parser.py)
# and Qt rendering (modules/overlay.py, modules/canvas.py) so they're plain,
# unit-testable functions.
#
# Two families of functions live here:
#   1. CSVL / apex vertebra -- used by the live clinical workspace
#      (modules/main_window.py via ScoliosisModelEngine).
#   2. Model validation / QA comparisons -- used by modules/validation.py
#      to compare a prediction against a ground-truth label file. This is a
#      different audience (ML team benchmarking the model) from the
#      clinical workspace, which never has a ground-truth label at
#      inference time for a real patient.

import math


# ---------------------------------------------------------------------------
# Oblique angle (per-vertebra endplate angle feeding into the Cobb angle)
# ---------------------------------------------------------------------------
#
# This is a direct port of cal_oblique01() in server.py -- the real model's
# inference server -- NOT a generic atan2. It matters that these match
# exactly: an earlier version of this app recomputed oblique angles locally
# with math.atan2(dy, dx), which agrees with cal_oblique01 only when the
# "left" keypoint's x-coordinate is less than the "right" keypoint's
# (dx >= 0). Verified by exhaustive comparison across dx/dy combinations:
# whenever a vertebra is rotated enough that the two keypoints' x-order
# flips (dx < 0) -- plausible right at a curve's apex, exactly where an
# accurate Cobb angle matters most -- atan2 and cal_oblique01 diverge by up
# to 180 degrees. Keeping this in lockstep with server.py is required both
# for the initial load (ScoliosisModelEngine.recalculate_all_metrics
# overwrites the server's own oblique/cobb values using this function) and
# for local recalculation after a clinician drags a keypoint in Edit Mode.

def oblique_angle(p1, p2):
    """Oblique angle in degrees between two endplate corner points, using
    the same convention as server.py's cal_oblique01 (bounded roughly to
    +/-90 degrees for a near-horizontal line, independent of which point is
    labeled "left" vs "right")."""
    x1_, y1_ = p1[0], p1[1]
    x2_, y2_ = p2[0], p2[1]
    x_x = x1_ - x2_
    y_y = y1_ - y2_
    if x_x == 0 and y_y == 0:
        return 0.0
    if x_x == 0 and y_y > 0:
        return -90.0
    if x_x == 0 and y_y < 0:
        return 90.0
    if x_x < 0 and y_y == 0:
        return 0.0
    if x_x > 0 and y_y == 0:
        return 0.0
    if (x_x < 0 and y_y > 0) or (x_x < 0 and y_y < 0):
        return float(math.degrees(math.atan(y_y / x_x)))
    if x_x > 0 and y_y > 0:
        return float(-90.0 - math.degrees(math.atan(y_y / x_x)))
    if x_x > 0 and y_y < 0:
        return float(90.0 + math.degrees(math.atan(y_y / x_x)))
    return 0.0


def cobb_angle_between_obliques(first_oblique, second_oblique):
    """Return the acute Cobb angle between two endplate orientations.

    Endplates are lines rather than directed vectors, so orientations that
    differ by 180 degrees describe the same line.  Normalize that periodic
    difference before choosing the smaller of the two intersecting angles.
    """
    difference = abs(float(second_oblique) - float(first_oblique)) % 180.0
    return min(difference, 180.0 - difference)


# ---------------------------------------------------------------------------
# CSVL (Central Sacral Vertical Line) and apex vertebra
# ---------------------------------------------------------------------------
#
# The backend JSON (see test_api_visualization.ipynb) doesn't label any
# detection as "sacrum" -- there's a "class" field but checking actual
# coordinates shows the one detection with class=1 sits near the TOP of the
# spine, not the bottom, so it isn't a sacral marker. Absent an explicit
# label, the bottommost detected vertebra (by vertical position) is used as
# the CSVL reference instead -- anatomically the closest available proxy to
# the sacral level. Deviations are reported in pixels: the JSON carries no
# pixel-spacing/calibration field, so there's no way to convert to mm.

def bottommost_detection_index(detections):
    """Index (list position) of the vertebra closest to the bottom of the
    image, i.e. the largest keypoint-0 (center) y-coordinate."""
    if not detections:
        return None
    return max(range(len(detections)), key=lambda i: detections[i]["keypoints"][0][1])


def compute_csvl_x(detections):
    """X-coordinate of the CSVL reference line (center-x of the bottommost
    detected vertebra)."""
    idx = bottommost_detection_index(detections)
    if idx is None:
        return None
    return detections[idx]["keypoints"][0][0]


def compute_apex(detections, csvl_x):
    """Returns (apex_index, deviation_px). The apex vertebra is defined,
    per standard scoliosis convention, as the vertebra whose center
    deviates furthest horizontally from the CSVL."""
    if not detections or csvl_x is None:
        return None, 0.0
    best_idx, best_dev = None, -1.0
    for i, det in enumerate(detections):
        dev = abs(det["keypoints"][0][0] - csvl_x)
        if dev > best_dev:
            best_idx, best_dev = i, dev
    return best_idx, best_dev


# ---------------------------------------------------------------------------
# Model validation / QA comparisons (prediction vs. ground-truth label)
# ---------------------------------------------------------------------------

def compare_detection_counts(pred_detections, label_detections):
    """1.1 -- does the model find the right number of vertebrae?
    Fewer than label suggests an undertrained model / weak architecture;
    more than label suggests overfitting / duplicated features."""
    pred_n, label_n = len(pred_detections), len(label_detections)
    diff = pred_n - label_n
    if diff == 0:
        verdict = "Match"
    elif diff < 0:
        verdict = f"{abs(diff)} missing vs. label (possible underfit / weak architecture)"
    else:
        verdict = f"{diff} extra vs. label (possible overfit / duplicated features)"
    return {"predicted": pred_n, "label": label_n, "diff": diff, "verdict": verdict}


def compare_oblique_angles(pred_detections, label_detections):
    """1.2 -- per-vertebra upper/lower oblique angle error, matched by list
    index. These angles feed directly into the Cobb angle calculation, so
    error here explains error downstream."""
    n = min(len(pred_detections), len(label_detections))
    upper_errors, lower_errors = [], []
    rows = []
    for i in range(n):
        p, l = pred_detections[i], label_detections[i]
        u_err = abs(p.get("upper_oblique", 0.0) - l.get("upper_oblique", 0.0))
        l_err = abs(p.get("lower_oblique", 0.0) - l.get("lower_oblique", 0.0))
        upper_errors.append(u_err)
        lower_errors.append(l_err)
        rows.append({"index": i, "upper_error": u_err, "lower_error": l_err})

    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "rows": rows,
        "mean_upper_error": _mean(upper_errors),
        "mean_lower_error": _mean(lower_errors),
        "max_upper_error": max(upper_errors) if upper_errors else 0.0,
        "max_lower_error": max(lower_errors) if lower_errors else 0.0,
    }


def compare_cobb_angle_counts(pred_pairs, label_pairs):
    """2.1 (count) -- does the model find the right number of curves?
    Fewer than label: model may be missing a curve. More than label:
    possible overfitting."""
    pred_n, label_n = len(pred_pairs), len(label_pairs)
    diff = pred_n - label_n
    if diff == 0:
        verdict = "Match"
    elif diff < 0:
        verdict = f"{abs(diff)} fewer curve(s) than label (model may be missing a curve)"
    else:
        verdict = f"{diff} more curve(s) than label (possible overfitting)"
    return {"predicted": pred_n, "label": label_n, "diff": diff, "verdict": verdict}


def compare_cobb_angle_values(pred_pairs, label_pairs):
    """2.1 (accuracy) -- how close are the predicted Cobb angle degrees to
    the label, curve by curve."""
    n = min(len(pred_pairs), len(label_pairs))
    rows, errors = [], []
    for i in range(n):
        p_angle = pred_pairs[i].get("cobb_angle", 0.0)
        l_angle = label_pairs[i].get("cobb_angle", 0.0)
        err = abs(p_angle - l_angle)
        errors.append(err)
        rows.append({"pair_index": i, "predicted": p_angle, "label": l_angle, "error": err})
    mean_err = sum(errors) / len(errors) if errors else 0.0
    max_err = max(errors) if errors else 0.0
    return {"rows": rows, "mean_error": mean_err, "max_error": max_err}
