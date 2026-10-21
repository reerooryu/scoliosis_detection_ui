# Tests for modules/geometry.py: endplate tilt, Cobb angle, curve choice,
# CSVL and apex. Pure math, so no Qt, GPU or Detectron2 is needed. The
# sample spine is made up (tests/synthetic_spine.py).

import math
import random

import pytest

from modules.geometry import (
    MIN_COBB_ANGLE, MIN_CURVE_SPAN,
    cobb_angle_between_obliques, compute_apex, compute_csvl_x,
    curve_apex, curve_start_index, find_cobb_curves, oblique_angle, select_angle_pairs,
    vertebra_levels,
)

from synthetic_spine import EXPECTED_CURVES, MARKER_INDEX, TILTS, make_result


def load_sample():
    return make_result()


def tilts(detections):
    upper = [oblique_angle(d["keypoints"][1], d["keypoints"][2]) for d in detections]
    lower = [oblique_angle(d["keypoints"][3], d["keypoints"][4]) for d in detections]
    return upper, lower


def endplate(angle_deg, length=100.0, start=(10.0, 20.0)):
    """Two corners of an endplate tilted by angle_deg."""
    x, y = start
    a = math.radians(angle_deg)
    return (x, y), (x + length * math.cos(a), y + length * math.sin(a))


# ---------------------------------------------------------------------------
# Endplate tilt
# ---------------------------------------------------------------------------

def test_flat_endplate_has_zero_tilt():
    assert oblique_angle((0, 0), (100, 0)) == 0.0


def test_right_corner_lower_is_positive():
    # Image y grows downward, so "lower" means a larger y.
    assert oblique_angle((0, 0), (100, 100)) == pytest.approx(45.0)
    assert oblique_angle((0, 100), (100, 0)) == pytest.approx(-45.0)


def test_vertical_and_zero_length_endplates():
    assert oblique_angle((0, 0), (0, 50)) == 90.0
    assert oblique_angle((5, 5), (5, 5)) == 0.0


@pytest.mark.parametrize("angle", range(-85, 90, 5))
def test_tilt_is_the_endplate_angle(angle):
    p1, p2 = endplate(angle)
    assert oblique_angle(p1, p2) == pytest.approx(angle)


@pytest.mark.parametrize("angle", range(-85, 90, 5))
def test_corner_order_does_not_matter(angle):
    p1, p2 = endplate(angle)
    assert oblique_angle(p2, p1) == pytest.approx(oblique_angle(p1, p2))


@pytest.mark.parametrize("angle", [0, 3, -7, 25, -40])
def test_swapped_corners_give_no_false_cobb_angle(angle):
    # The old formula gave about 78 degrees for a 5.7 degree endplate
    # measured against itself with its corners swapped.
    p1, p2 = endplate(angle)
    assert cobb_angle_between_obliques(oblique_angle(p1, p2), oblique_angle(p2, p1)) == pytest.approx(0.0, abs=1e-9)


def _original_formula(p1, p2):
    """The server's old formula (cal_oblique01), kept only to show that the
    new one gives the same answer whenever the corners are in order."""
    x_x = p1[0] - p2[0]
    y_y = p1[1] - p2[1]
    if x_x == 0 and y_y == 0:
        return 0.0
    if x_x == 0:
        return -90.0 if y_y > 0 else 90.0
    if y_y == 0:
        return 0.0
    if x_x < 0:
        return math.degrees(math.atan(y_y / x_x))
    if y_y > 0:
        return -90.0 - math.degrees(math.atan(y_y / x_x))
    return 90.0 + math.degrees(math.atan(y_y / x_x))


def test_same_as_original_formula_when_corners_are_in_order():
    rng = random.Random(0)
    for _ in range(10000):
        p1 = (rng.uniform(0, 2000), rng.uniform(0, 4000))
        p2 = (p1[0] + rng.uniform(1, 200), p1[1] + rng.uniform(-150, 150))
        assert oblique_angle(p1, p2) == pytest.approx(_original_formula(p1, p2), abs=1e-9)


def test_sample_tilts_match_the_designed_tilts():
    upper, lower = tilts(load_sample()["detections"])
    assert upper == pytest.approx(TILTS, abs=1e-9)
    assert lower == pytest.approx(TILTS, abs=1e-9)


# ---------------------------------------------------------------------------
# Cobb angle between two tilts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("first, second, expected", [
    (10, -20, 30),
    (0, 0, 0),
    (0, 90, 90),
    (-80, 80, 20),     # endplates are lines: 160 degrees apart is 20 degrees
    (45, -45, 90),
])
def test_cobb_angle(first, second, expected):
    assert cobb_angle_between_obliques(first, second) == pytest.approx(expected)
    assert cobb_angle_between_obliques(second, first) == pytest.approx(expected)


def test_cobb_angle_is_between_0_and_90():
    rng = random.Random(1)
    for _ in range(1000):
        angle = cobb_angle_between_obliques(rng.uniform(-90, 90), rng.uniform(-90, 90))
        assert 0.0 <= angle <= 90.0


# ---------------------------------------------------------------------------
# Curve selection
# ---------------------------------------------------------------------------

def test_sample_curves():
    detections = load_sample()["detections"]
    upper, lower = tilts(detections)
    start = curve_start_index([d["class"] for d in detections])
    assert start == MARKER_INDEX
    curves = find_cobb_curves(upper, lower, start)
    assert [(u, l) for u, l, _ in curves] == [(u, l) for u, l, _ in EXPECTED_CURVES]
    assert [a for _, _, a in curves] == pytest.approx([a for _, _, a in EXPECTED_CURVES], abs=1e-6)


def test_curve_rules_hold_on_sample():
    detections = load_sample()["detections"]
    upper, lower = tilts(detections)
    start = curve_start_index([d["class"] for d in detections])
    curves = find_cobb_curves(upper, lower, start)
    for u, l, angle in curves:
        assert angle > MIN_COBB_ANGLE
        assert l - u >= MIN_CURVE_SPAN
        assert u >= start
    for (u1, l1, _), (u2, l2, _) in zip(curves, curves[1:]):
        assert l1 == u2                                    # shared end vertebra
        bend1 = lower[l1] - upper[u1]
        bend2 = lower[l2] - upper[u2]
        assert bend1 * bend2 < 0                           # opposite directions


def test_straight_spine_has_no_curve():
    rng = random.Random(2)
    upper = [rng.uniform(-3, 3) for _ in range(17)]
    lower = [rng.uniform(-3, 3) for _ in range(17)]
    assert find_cobb_curves(upper, lower) == []


def test_vertebrae_above_the_marker_are_ignored():
    # A strong bend only in vertebrae 0-2; everything below is flat.
    upper = [30, 0, 0, 0, 0, 0, 0, 0]
    lower = [0, 0, -30, 0, 0, 0, 0, 0]
    assert find_cobb_curves(upper, lower, start_index=0) != []
    assert find_cobb_curves(upper, lower, start_index=3) == []


def test_curve_start_index():
    assert curve_start_index([0, 0, 1, 0]) == 2
    assert curve_start_index([0, 0, 0]) == 0


def test_select_angle_pairs_format():
    detections = load_sample()["detections"]
    upper, lower = tilts(detections)
    pairs = select_angle_pairs(upper, lower, 5)
    assert [p["pair_index"] for p in pairs] == list(range(len(pairs)))
    for p in pairs:
        assert p["upper_oblique"] == upper[p["upper_detection_index"]]
        assert p["lower_oblique"] == lower[p["lower_detection_index"]]
        assert p["cobb_angle"] == pytest.approx(cobb_angle_between_obliques(p["upper_oblique"], p["lower_oblique"]))


# ---------------------------------------------------------------------------
# Level names
# ---------------------------------------------------------------------------

def test_levels_are_counted_up_from_l5():
    levels = vertebra_levels(23)      # the sample spine: C2 at the top
    assert levels[-1] == "L5"
    assert levels[18] == "L1"
    assert levels[17] == "T12"
    assert levels[6] == "T1"
    assert levels[MARKER_INDEX] == "C7"
    assert levels[0] == "C2"


def test_full_spine_and_extra_vertebrae():
    assert vertebra_levels(24)[0] == "C1"
    assert vertebra_levels(25)[:2] == [None, "C1"]    # above C1: no name
    assert vertebra_levels(3) == ["L3", "L4", "L5"]   # only the lumbar spine shown
    assert vertebra_levels(0) == []


# ---------------------------------------------------------------------------
# CSVL and apex
# ---------------------------------------------------------------------------

def _det(cx, cy):
    return {"keypoints": [[cx, cy, 1.0]] + [[cx, cy, 1.0]] * 4}


def test_csvl_runs_through_the_lowest_vertebra():
    detections = [_det(100, 10), _det(180, 50), _det(120, 90)]
    assert compute_csvl_x(detections) == 120


def test_apex_is_furthest_from_the_csvl():
    detections = [_det(100, 10), _det(180, 50), _det(120, 90)]
    assert compute_apex(detections, 120) == (1, 60)


def test_curve_apex_is_furthest_from_the_end_vertebrae_line():
    # End vertebrae at x=0; the middle one bulges furthest.
    detections = [_det(0, 0), _det(10, 10), _det(25, 20), _det(12, 30), _det(0, 40)]
    assert curve_apex(detections, 0, 4) == (2, pytest.approx(25.0))


def test_curve_apex_is_never_an_end_vertebra():
    # The lower end vertebra is far off to the side (as next to a larger
    # curve), but the apex must lie between the ends.
    detections = [_det(0, 0), _det(5, 10), _det(8, 20), _det(200, 30)]
    apex, _ = curve_apex(detections, 0, 3)
    assert apex in (1, 2)


def test_curve_apex_of_each_sample_curve_lies_inside_it():
    detections = load_sample()["detections"]
    for u, l, _ in EXPECTED_CURVES:
        apex, distance = curve_apex(detections, u, l)
        assert u < apex < l
        assert distance > 0


def test_whole_spine_apex_wins_inside_its_curve():
    detections = [_det(0, 0), _det(10, 10), _det(25, 20), _det(12, 30), _det(0, 40)]
    assert curve_apex(detections, 0, 4, main_apex=3)[0] == 3     # inside: used
    assert curve_apex(detections, 0, 4, main_apex=4)[0] == 2     # an end vertebra: ignored
    assert curve_apex(detections, 0, 4, main_apex=None)[0] == 2


def test_curve_apex_needs_a_vertebra_between_the_ends():
    detections = [_det(0, 0), _det(5, 10)]
    assert curve_apex(detections, 0, 1) == (None, 0.0)


def test_csvl_and_apex_with_no_detections():
    assert compute_csvl_x([]) is None
    assert compute_apex([], None) == (None, 0.0)
