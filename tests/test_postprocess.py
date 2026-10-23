# Tests for modules/postprocess.py: the steps between the model's raw output
# and the result JSON (scaling, filtering, sorting). numpy only.

import numpy as np
import pytest

from modules.postprocess import SCORE_THRESHOLD, filter_and_sort_detections, scale_prediction_data


def test_scaling_maps_boxes_and_keypoints_to_the_original_size():
    boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
    keypoints = np.array([[[5, 10, 0.9], [1, 2, 0.5], [3, 4, 0.5], [6, 7, 0.5], [8, 9, 0.5]]], dtype=np.float32)
    # Model input 768 x 1536 (w x h); original 1536 x 768: x doubles, y halves.
    b, k = scale_prediction_data(boxes, keypoints, original_shape=(768, 1536, 3), model_shape=(1536, 768, 3))
    assert b.tolist() == [[20, 10, 60, 20]]
    assert k[0, 0].tolist() == pytest.approx([10, 5, 0.9])     # confidence unchanged
    assert b.dtype == float and k.dtype == float


def test_scaling_with_no_detections():
    b, k = scale_prediction_data(np.zeros((0, 4)), np.zeros((0, 5, 3)), (1000, 500, 3), (1536, 768, 3))
    assert b.shape == (0, 4) and k.shape == (0, 5, 3)


def _detections(rows):
    """rows: (class, score, center_y). Returns arrays as the model gives them."""
    classes = np.array([r[0] for r in rows])
    scores = np.array([r[1] for r in rows], dtype=np.float32)
    boxes = np.array([[0, r[2] - 5, 10, r[2] + 5] for r in rows], dtype=float)
    keypoints = np.zeros((len(rows), 5, 3))
    return classes, scores, boxes, keypoints


def test_filter_drops_low_scores_and_sorts_top_to_bottom():
    classes, scores, boxes, keypoints = _detections([(0, 0.95, 300), (0, 0.5, 100), (0, 0.9, 50)])
    c, s, b, _ = filter_and_sort_detections(classes, scores, boxes, keypoints)
    assert [round(x, 2) for x in s] == [0.9, 0.95]            # 0.5 dropped; sorted by height
    assert [box[1] for box in b] == [45, 295]


def test_score_exactly_at_the_threshold_is_kept():
    classes, scores, boxes, keypoints = _detections([(0, SCORE_THRESHOLD, 100)])
    assert len(filter_and_sort_detections(classes, scores, boxes, keypoints)[0]) == 1


def test_only_the_most_confident_marker_vertebra_is_kept():
    classes, scores, boxes, keypoints = _detections([(1, 0.85, 100), (0, 0.9, 200), (1, 0.97, 150)])
    c, s, _, _ = filter_and_sort_detections(classes, scores, boxes, keypoints)
    assert c.count(1) == 1
    assert s[c.index(1)] == pytest.approx(0.97)
