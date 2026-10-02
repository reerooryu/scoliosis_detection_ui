# Clinical geometry: plain math with no Qt and no file or network access.
#   1. Endplate tilt, Cobb angle, CSVL and apex -- used by the workspace.
#   2. Prediction-vs-label comparisons -- used by the Model Validation tool.

import math


# ---------------------------------------------------------------------------
# Endplate tilt ("oblique angle") and Cobb angle
# ---------------------------------------------------------------------------
#
# oblique_angle() must give exactly the same results as cal_oblique01() in
# server.py. It is not a plain atan2: the two differ when the "left" corner
# lies to the right of the "right" corner. The app recalculates every angle
# with this function, so any difference would change the reported angles.

def oblique_angle(p1, p2):
    """Endplate tilt in degrees from corner p1 (left) to corner p2 (right).
    Same rules as cal_oblique01 in server.py."""
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
    """Cobb angle in degrees (0 to 90) between two endplate tilts.

    An endplate is a line, not an arrow, so tilts 180 degrees apart are the
    same line. The smaller of the two crossing angles is returned.
    """
    difference = abs(float(second_oblique) - float(first_oblique)) % 180.0
    return min(difference, 180.0 - difference)


# ---------------------------------------------------------------------------
# CSVL (Central Sacral Vertical Line) and apex vertebra
# ---------------------------------------------------------------------------
#
# The server does not label the sacrum, so the lowest detected vertebra is
# used as the CSVL reference. Deviations are in pixels: the result has no
# pixel spacing, so they cannot be converted to mm.

def bottommost_detection_index(detections):
    """List position of the lowest vertebra in the image (largest center y)."""
    if not detections:
        return None
    return max(range(len(detections)), key=lambda i: detections[i]["keypoints"][0][1])


def compute_csvl_x(detections):
    """X position of the CSVL: the center x of the lowest vertebra."""
    idx = bottommost_detection_index(detections)
    if idx is None:
        return None
    return detections[idx]["keypoints"][0][0]


def compute_apex(detections, csvl_x):
    """Returns (apex_index, deviation_px): the vertebra whose center is
    furthest sideways from the CSVL."""
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
    """Compare the number of vertebrae found against the label."""
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
    """Upper and lower tilt error for each vertebra. Vertebrae are matched by
    list position, so a missing or extra detection shifts every row after it."""
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
    """Compare the number of Cobb curves found against the label."""
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
    """Cobb angle error for each curve, matched by list position."""
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
