# Turns the model's raw output into the result the server sends back: scale
# it to the original image, filter and sort the detections, then work out the
# tilts and curves. Plain numpy and math only -- no Detectron2, PyTorch or
# OpenCV -- so it runs and can be tested without the model.
#
# server.py runs the model, then calls these functions. Masks are not handled
# here: the result does not use them.

from typing import Dict, List, Optional

import numpy as np

from modules.geometry import curve_start_index, oblique_angle, select_angle_pairs


def scale_prediction_data(boxes, keypoints, original_shape, model_shape):
    """Map boxes and keypoints from model-input size back to the original
    image size."""
    h0, w0 = original_shape[:2]
    h1, w1 = model_shape[:2]
    h_ratio = h0 / h1
    w_ratio = w0 / w1

    scaled_boxes = []
    scaled_keypoints = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        scaled_boxes.append([x1 * w_ratio, y1 * h_ratio, x2 * w_ratio, y2 * h_ratio])

        keypoints_i = []
        for kp in keypoints[i]:
            keypoints_i.append([kp[0] * w_ratio, kp[1] * h_ratio, float(kp[2])])
        scaled_keypoints.append(np.asarray(keypoints_i, dtype=float))

    return np.asarray(scaled_boxes, dtype=float), np.asarray(scaled_keypoints, dtype=float)


def filter_and_sort_detections(classes, scores, boxes, keypoints, score_threshold=0.8):
    """Keep detections scoring at least score_threshold, keep only the best
    class-1 detection, and sort them top to bottom."""
    pc0 = []
    pc1 = []
    for i, cls in enumerate(classes):
        entry = {
            "class": int(cls),
            "score": float(scores[i]),
            "box": boxes[i].tolist(),
            "keypoints": keypoints[i],
        }
        if cls == 0 and scores[i] >= score_threshold:
            pc0.append(entry)
        elif cls == 1 and scores[i] >= score_threshold:
            pc1.append(entry)

    if len(pc1) > 1:
        # Keep the most confident one, not just the first one found.
        pc1 = [max(pc1, key=lambda e: e["score"])]

    joint = pc0 + pc1
    if not joint:
        return [], [], [], []

    joint = sorted(joint, key=lambda x: (x["box"][1] + x["box"][3]) / 2)
    classes_sorted = [item["class"] for item in joint]
    scores_sorted = [item["score"] for item in joint]
    boxes_sorted = [item["box"] for item in joint]
    keypoints_sorted = [item["keypoints"] for item in joint]

    return classes_sorted, scores_sorted, boxes_sorted, keypoints_sorted


def compute_cobb_results(
    classes_list: List[int],
    boxes: List[List[float]],
    keypoints: List[np.ndarray],
    scores_list: Optional[List[float]] = None,
):
    """Compute each vertebra's endplate tilts, then the curves and their Cobb
    angles. The curve rules live in modules/geometry.py:find_cobb_curves."""
    results: Dict[str, object] = {
        "all_angles": [],
        "selected_cobb_angle": None,
        "upper_obliques": [],
        "lower_obliques": [],
        "end_vertebrae_indices": [],
        "masker_point_upper": [],
        "masker_point_lower": [],
        "detections": [],
        "angle_pairs": [],
    }

    if not classes_list:
        return results

    degree_upper = []
    degree_lower = []
    for i, kp in enumerate(keypoints):
        degree_upper.append(oblique_angle(kp[1], kp[2]))
        degree_lower.append(oblique_angle(kp[3], kp[4]))
        results["detections"].append(
            {
                "index": i,
                "class": int(classes_list[i]),
                "score": float(scores_list[i]) if scores_list is not None else None,
                "box": boxes[i],
                "upper_oblique": float(degree_upper[-1]),
                "lower_oblique": float(degree_lower[-1]),
                "keypoints": kp.tolist(),
            }
        )

    results["upper_obliques"] = [float(x) for x in degree_upper]
    results["lower_obliques"] = [float(x) for x in degree_lower]

    # Curves: every curve above 10 degrees, largest first, neighbours sharing
    # an end vertebra, searched from the marker vertebra downward.
    angle_pairs = select_angle_pairs(degree_upper, degree_lower, curve_start_index(classes_list))
    cobb_angles = [pair["cobb_angle"] for pair in angle_pairs]
    upper_ends = [pair["upper_detection_index"] for pair in angle_pairs]
    lower_ends = [pair["lower_detection_index"] for pair in angle_pairs]

    results["angle_pairs"] = angle_pairs
    results["all_angles"] = cobb_angles
    results["selected_cobb_angle"] = float(max(cobb_angles)) if cobb_angles else None
    results["end_vertebrae_indices"] = sorted(set(upper_ends + lower_ends))
    # Older names, kept so existing notebooks keep working.
    results["masker_point_upper"] = upper_ends
    results["masker_point_lower"] = lower_ends

    return results
