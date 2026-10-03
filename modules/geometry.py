# Clinical geometry: plain math with no Qt and no file or network access.
#   1. Endplate tilt, Cobb angle and curve selection -- used by the server
#      and by the workspace, so both follow the same rules.
#   2. CSVL and apex -- used by the workspace.
#   3. Prediction-vs-label comparisons -- used by the Model Validation tool.

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
# Curve selection: which vertebrae are the end vertebrae of each curve
# ---------------------------------------------------------------------------
#
# server.py uses this for the first result, and the app uses it again after
# a landmark is edited, so the curves always follow the same rules.

MIN_COBB_ANGLE = 10.0   # a curve counts only if its Cobb angle is above this
MIN_CURVE_SPAN = 2      # the two end vertebrae are at least this many places apart
MARKER_CLASS = 1        # detection class of the marker vertebra


def curve_start_index(classes):
    """List position of the marker vertebra (class 1), or 0 if there is none.

    Curves are searched from the marker downward. The vertebrae above it
    (the neck) are left out: their endplates are small, so their tilts are
    too unreliable to measure curves from.
    """
    for index, cls in enumerate(classes):
        if cls == MARKER_CLASS:
            return index
    return 0


def find_cobb_curves(upper_tilts, lower_tilts, start_index=0, min_angle=MIN_COBB_ANGLE):
    """Choose the end vertebrae of every curve. Returns a list of
    (upper_index, lower_index, cobb_angle), ordered top to bottom.

    A curve is measured from the upper endplate of its upper end vertebra to
    the lower endplate of its lower end vertebra. The rules:
      - a curve counts only if its Cobb angle is more than min_angle;
      - a curve is a single bend: no curve bending the other way lies
        between its end vertebrae;
      - neighbouring curves share an end vertebra: the lower end vertebra of
        one curve is the upper end vertebra of the next;
      - neighbouring curves bend in opposite directions.

    The largest curve is found first, from every possible pair of vertebrae.
    Curves are then added above and below it one at a time. Each starts at
    the shared end vertebra and takes the other end vertebra that gives it
    the largest angle.
    """
    count = min(len(upper_tilts), len(lower_tilts))

    # Every pair of vertebrae that could be a curve: (angle, bend direction).
    pairs = {}
    for upper_index in range(start_index, count):
        for lower_index in range(upper_index + MIN_CURVE_SPAN, count):
            angle = cobb_angle_between_obliques(upper_tilts[upper_index], lower_tilts[lower_index])
            if angle > min_angle:
                direction = 1 if lower_tilts[lower_index] > upper_tilts[upper_index] else -1
                pairs[(upper_index, lower_index)] = (angle, direction)

    def is_single_bend(upper_index, lower_index):
        """False if a pair bending the other way lies inside this one. The
        pair then spans several curves, so it is not one curve."""
        direction = pairs[(upper_index, lower_index)][1]
        return not any(
            other_direction == -direction and upper_index <= other_upper and other_lower <= lower_index
            for (other_upper, other_lower), (_, other_direction) in pairs.items()
        )

    def largest(keep):
        """The single-bend pair with the largest angle among those that
        keep(upper_index, lower_index, direction) accepts, or None."""
        best = None
        for (upper_index, lower_index), (angle, direction) in pairs.items():
            if not keep(upper_index, lower_index, direction):
                continue
            if (best is None or angle > best[2]) and is_single_bend(upper_index, lower_index):
                best = (upper_index, lower_index, angle)
        return best

    main_curve = largest(lambda upper_index, lower_index, direction: True)
    if main_curve is None:
        return []
    curves = [main_curve]

    # Curves above: each ends on the upper end vertebra of the curve below it.
    while True:
        shared = curves[0][0]
        wanted = -pairs[curves[0][:2]][1]
        above = largest(lambda upper_index, lower_index, direction: lower_index == shared and direction == wanted)
        if above is None:
            break
        curves.insert(0, above)

    # Curves below: each starts on the lower end vertebra of the curve above it.
    while True:
        shared = curves[-1][1]
        wanted = -pairs[curves[-1][:2]][1]
        below = largest(lambda upper_index, lower_index, direction: upper_index == shared and direction == wanted)
        if below is None:
            break
        curves.append(below)

    return curves


def select_angle_pairs(upper_tilts, lower_tilts, start_index=0):
    """The curves from find_cobb_curves as "angle_pairs" entries, the form
    used in the server result and by the app."""
    pairs = []
    for upper_index, lower_index, angle in find_cobb_curves(upper_tilts, lower_tilts, start_index):
        pairs.append({
            "pair_index": len(pairs),
            "upper_detection_index": upper_index,
            "lower_detection_index": lower_index,
            "upper_oblique": float(upper_tilts[upper_index]),
            "lower_oblique": float(lower_tilts[lower_index]),
            "cobb_angle": angle,
        })
    return pairs


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
