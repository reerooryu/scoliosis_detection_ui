import io
import math
from threading import Lock
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import timm
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from modules.geometry import cobb_angle_between_obliques

app = FastAPI(title="Cobb Angle Inference API")
ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT_DIR / "model" / "model_t001_6_effb5_mask_kp_2cls" / "model_final_run.pth"

predictor = None
# FastAPI executes synchronous endpoints in its worker threadpool.  Keep
# model construction and predictor use serialized: Detectron2 predictor
# initialization is expensive and concurrent GPU calls on one shared model
# can cause duplicate allocations or unsafe execution.
_predictor_lock = Lock()
_inference_lock = Lock()


@BACKBONE_REGISTRY.register()
def build_efficientnet_b5_fpn(cfg, input_shape: ShapeSpec):
    body = timm.create_model(
        "efficientnet_b5",
        pretrained=False,
        features_only=True,
        out_indices=(1, 2, 3, 4),
    )

    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        body = body.to(device)
        feats = body(torch.zeros(1, 3, 256, 256, device=device))
        in_channels_list = [f.shape[1] for f in feats]
        body = body.to("cpu")

    class TimmBackbone(Backbone):
        def __init__(self, body, in_channels_list):
            super().__init__()
            self.body = body
            self._out_features = ["res2", "res3", "res4", "res5"]
            self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
            self._out_feature_channels = dict(zip(self._out_features, in_channels_list))

        def forward(self, x):
            feats = self.body(x)
            return {name: feats[i] for i, name in enumerate(self._out_features)}

        def output_shape(self):
            return {
                k: ShapeSpec(channels=self._out_feature_channels[k], stride=self._out_feature_strides[k])
                for k in self._out_features
            }

    bottom_up = TimmBackbone(body, in_channels_list)
    return FPN(
        bottom_up=bottom_up,
        in_features=bottom_up._out_features,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm="",
        top_block=LastLevelMaxPool(),
    )


def load_predictor():
    global predictor
    if predictor is not None:
        return predictor

    with _predictor_lock:
        # A second request can reach this point while the first is loading the
        # weights. Recheck while holding the lock to avoid allocating the
        # model (and its GPU memory) twice.
        if predictor is not None:
            return predictor

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = str(WEIGHTS_PATH)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.DATASETS.TEST = ("vbs_test",)
        cfg.INPUT.FORMAT = "RGB"
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53]
        cfg.MODEL.PIXEL_STD = [58.395, 57.12, 57.375]
        cfg.INPUT.MIN_SIZE_TEST = 768
        cfg.INPUT.MAX_SIZE_TEST = 1536
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.KEYPOINT_ON = True
        cfg.MODEL.BACKBONE.NAME = "build_efficientnet_b5_fpn"
        cfg.MODEL.FPN.OUT_CHANNELS = 256
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
        cfg.MODEL.ROI_MASK_HEAD.CONV_DIM = 512
        cfg.MODEL.ROI_MASK_HEAD.NUM_CONVS = 16
        cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.45
        cfg.TEST.DETECTIONS_PER_IMAGE = 300

        predictor = DefaultPredictor(cfg)
    return predictor


def cal_oblique01(x1_, y1_, x2_, y2_):
    x_x = x1_ - x2_
    y_y = y1_ - y2_
    if (x_x == 0) and (y_y == 0):
        return 0.0
    if (x_x == 0) and (y_y > 0):
        return -90.0
    if (x_x == 0) and (y_y < 0):
        return 90.0
    if (x_x < 0) and (y_y == 0):
        return 0.0
    if (x_x > 0) and (y_y == 0):
        return 0.0
    if ((x_x < 0) and (y_y > 0)) or ((x_x < 0) and (y_y < 0)):
        return float(math.degrees(math.atan(y_y / x_x)))
    if (x_x > 0) and (y_y > 0):
        return float(-90.0 - math.degrees(math.atan(y_y / x_x)))
    if (x_x > 0) and (y_y < 0):
        return float(90.0 + math.degrees(math.atan(y_y / x_x)))
    return 0.0


def savgol_smooth_1d(y, window=5, polyorder=2, mode="reflect"):
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.size
    if n == 0:
        return y.copy()

    if window % 2 == 0:
        window += 1
    if window < polyorder + 2:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
        window = max(window, polyorder + 2 + ((polyorder + 2) % 2 == 0))

    half = window // 2
    t = np.arange(-half, half + 1, dtype=float)
    A = np.vander(t, N=polyorder + 1, increasing=True)
    ATA_inv = np.linalg.pinv(A.T @ A)
    e0 = np.zeros(polyorder + 1, dtype=float)
    e0[0] = 1.0
    c = e0 @ ATA_inv @ A.T

    if half > 0:
        if mode == "reflect":
            ypad = np.r_[y[half:0:-1], y, y[-2:-half - 2:-1]]
        elif mode == "edge":
            ypad = np.r_[np.full(half, y[0]), y, np.full(half, y[-1])]
        else:
            raise ValueError("mode must be 'reflect' or 'edge'")
    else:
        ypad = y

    return np.convolve(ypad, c[::-1], mode="valid")


def count_curves_filtered(y_filt, prom=3.0, prom_window=2, min_dist=2):
    y = np.asarray(y_filt, dtype=float).reshape(-1)
    n = len(y)
    if n < 3:
        return {
            "n_upper_curves": 0,
            "n_lower_curves": 0,
            "peaks_idx": np.array([], dtype=int),
            "peaks_val": np.array([], dtype=float),
            "valleys_idx": np.array([], dtype=int),
            "valleys_val": np.array([], dtype=float),
        }

    dy = np.diff(y)
    s = np.sign(dy)
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    for i in range(len(s) - 2, -1, -1):
        if s[i] == 0:
            s[i] = s[i + 1]

    peaks_idx = np.where((s[:-1] > 0) & (s[1:] < 0))[0] + 1
    valleys_idx = np.where((s[:-1] < 0) & (s[1:] > 0))[0] + 1

    def _prom_filter(idx, kind="max"):
        keep = []
        for i in idx:
            left = y[max(0, i - prom_window):i]
            right = y[i + 1 : min(n, i + 1 + prom_window)]
            if len(left) == 0 or len(right) == 0:
                continue
            if kind == "max":
                baseline = max(np.min(left), np.min(right))
                p = y[i] - baseline
            else:
                baseline = min(np.max(left), np.max(right))
                p = baseline - y[i]
            if p >= prom:
                keep.append(i)
        return np.array(keep, dtype=int)

    peaks_idx = _prom_filter(peaks_idx, kind="max")
    valleys_idx = _prom_filter(valleys_idx, kind="min")

    def _min_dist(idx, kind="max"):
        if len(idx) == 0:
            return idx
        idx = np.sort(idx)
        out = [idx[0]]
        for i in idx[1:]:
            if i - out[-1] >= min_dist:
                out.append(i)
            else:
                if kind == "max" and y[i] > y[out[-1]]:
                    out[-1] = i
                if kind == "min" and y[i] < y[out[-1]]:
                    out[-1] = i
        return np.array(out, dtype=int)

    peaks_idx = _min_dist(peaks_idx, kind="max")
    valleys_idx = _min_dist(valleys_idx, kind="min")

    return {
        "n_upper_curves": int(len(peaks_idx)),
        "n_lower_curves": int(len(valleys_idx)),
        "peaks_idx": peaks_idx,
        "peaks_val": y[peaks_idx] if len(peaks_idx) else np.array([], dtype=float),
        "valleys_idx": valleys_idx,
        "valleys_val": y[valleys_idx] if len(valleys_idx) else np.array([], dtype=float),
    }


def extend_segment(p1, p2, extend=50):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < 1e-6:
        return p1, p2
    u = v / length
    return p1 - extend * u, p2 + extend * u


def read_image_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_image_for_model(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (768, 1536), interpolation=cv2.INTER_LINEAR)


def scale_prediction_data(
    boxes: np.ndarray,
    masks: np.ndarray,
    keypoints: np.ndarray,
    original_shape: tuple,
    model_shape: tuple,
):
    h0, w0 = original_shape[:2]
    h1, w1 = model_shape[:2]
    h_ratio = h0 / h1
    w_ratio = w0 / w1

    scaled_boxes = []
    scaled_masks = []
    scaled_keypoints = []

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        scaled_boxes.append([x1 * w_ratio, y1 * h_ratio, x2 * w_ratio, y2 * h_ratio])

        mask = masks[i].astype(np.uint8) * 255
        resized_mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_LINEAR)
        scaled_masks.append(resized_mask > 127)

        keypoints_i = []
        for kp in keypoints[i]:
            keypoints_i.append([kp[0] * w_ratio, kp[1] * h_ratio, float(kp[2])])
        scaled_keypoints.append(np.asarray(keypoints_i, dtype=float))

    return np.asarray(scaled_boxes, dtype=float), np.asarray(scaled_masks, dtype=bool), np.asarray(scaled_keypoints, dtype=float)


def filter_and_sort_detections(
    classes: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    keypoints: np.ndarray,
    score_threshold: float = 0.8,
):
    pc0 = []
    pc1 = []
    for i, cls in enumerate(classes):
        entry = {
            "class": int(cls),
            "score": float(scores[i]),
            "box": boxes[i].tolist(),
            "mask": masks[i],
            "keypoints": keypoints[i],
        }
        if cls == 0 and scores[i] >= score_threshold:
            pc0.append(entry)
        elif cls == 1 and scores[i] >= score_threshold:
            pc1.append(entry)

    if len(pc1) > 1:
        # Keep the highest-confidence candidate, not just whichever happened
        # to come first in detection order.
        pc1 = [max(pc1, key=lambda e: e["score"])]

    joint = pc0 + pc1
    if not joint:
        return [], [], [], [], []

    joint = sorted(joint, key=lambda x: (x["box"][1] + x["box"][3]) / 2)
    classes_sorted = [item["class"] for item in joint]
    scores_sorted = [item["score"] for item in joint]
    boxes_sorted = [item["box"] for item in joint]
    masks_sorted = [item["mask"] for item in joint]
    keypoints_sorted = [item["keypoints"] for item in joint]

    return classes_sorted, scores_sorted, boxes_sorted, masks_sorted, keypoints_sorted


def compute_cobb_results(
    classes_list: List[int],
    boxes: List[List[float]],
    masks: List[np.ndarray],
    keypoints: List[np.ndarray],
    scores_list: Optional[List[float]] = None,
):
    results: Dict[str, object] = {
        "all_angles": [],
        "selected_cobb_angle": None,
        "upper_obliques": [],
        "lower_obliques": [],
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
        degree_upper.append(cal_oblique01(kp[1][0], kp[1][1], kp[2][0], kp[2][1]))
        degree_lower.append(cal_oblique01(kp[3][0], kp[3][1], kp[4][0], kp[4][1]))
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

    diff_degree_upper = np.concatenate(([0.0], np.diff(degree_upper))).tolist()
    diff_degree_lower = np.concatenate(([0.0], np.diff(degree_lower))).tolist()

    degree_upper_sg = savgol_smooth_1d(degree_upper, window=2, polyorder=3)
    degree_lower_sg = savgol_smooth_1d(degree_lower, window=2, polyorder=3)
    diff_upper_sg = savgol_smooth_1d(diff_degree_upper, window=2, polyorder=3)
    diff_lower_sg = savgol_smooth_1d(diff_degree_lower, window=2, polyorder=3)

    upper_ref = count_curves_filtered(degree_upper_sg, prom=3.0, prom_window=3, min_dist=3)
    lower_ref = count_curves_filtered(degree_lower_sg, prom=3.0, prom_window=3, min_dist=3)

    ths_cut_angle_peak = float(np.std(degree_upper_sg))
    reference_upper_positions = []
    if upper_ref["n_upper_curves"] + upper_ref["n_lower_curves"] > 0:
        positions = np.concatenate((upper_ref["peaks_idx"], upper_ref["valleys_idx"]))
        values = np.concatenate((upper_ref["peaks_val"], upper_ref["valleys_val"]))
        for pos, val in zip(positions, values):
            if abs(val) > ths_cut_angle_peak:
                reference_upper_positions.append(int(pos))

    reference_lower_positions = []
    if lower_ref["n_upper_curves"] + lower_ref["n_lower_curves"] > 0:
        positions = np.concatenate((lower_ref["peaks_idx"], lower_ref["valleys_idx"]))
        values = np.concatenate((lower_ref["peaks_val"], lower_ref["valleys_val"]))
        for pos, val in zip(positions, values):
            if abs(val) > ths_cut_angle_peak:
                reference_lower_positions.append(int(pos))

    masker_point_upper = [i for i, cls in enumerate(classes_list) if cls == 1]
    masker_point_lower = [i for i, cls in enumerate(classes_list) if cls == 1]
    masker_point_upper = sorted(set(masker_point_upper + reference_upper_positions))
    masker_point_lower = sorted(set(masker_point_lower + reference_lower_positions))

    results["masker_point_upper"] = masker_point_upper
    results["masker_point_lower"] = masker_point_lower
    results["upper_obliques"] = [float(x) for x in degree_upper]
    results["lower_obliques"] = [float(x) for x in degree_lower]

    cobb_angles = []
    angle_pairs = []
    for idx in range(len(masker_point_upper) - 1):
        upper_idx = masker_point_upper[idx]
        lower_idx = masker_point_upper[idx + 1]
        upper_angle = degree_upper[upper_idx]
        lower_angle = degree_lower[lower_idx]
        cobb_angle_value = cobb_angle_between_obliques(upper_angle, lower_angle)
        cobb_angles.append(cobb_angle_value)
        angle_pairs.append(
            {
                "pair_index": idx,
                "upper_detection_index": upper_idx,
                "lower_detection_index": lower_idx,
                "upper_oblique": float(upper_angle),
                "lower_oblique": float(lower_angle),
                "cobb_angle": cobb_angle_value,
            }
        )

    results["all_angles"] = cobb_angles
    results["angle_pairs"] = angle_pairs
    results["selected_cobb_angle"] = float(max(cobb_angles)) if cobb_angles else None

    return results


def run_inference(image_bytes: bytes) -> Dict[str, object]:
    image = read_image_bytes(image_bytes)
    image_model = prepare_image_for_model(image)
    predictor = load_predictor()
    # One shared predictor is intentionally serialized. Requests can still
    # wait here in FastAPI's threadpool without blocking the async event loop
    # or /health.
    with _inference_lock:
        outputs = predictor(image_model)
    instances = outputs["instances"].to("cpu")

    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else np.zeros((0, 4), dtype=float)
    masks = instances.pred_masks.numpy() if instances.has("pred_masks") else np.zeros((0, image_model.shape[0], image_model.shape[1]), dtype=bool)
    keypoints = instances.pred_keypoints.numpy() if instances.has("pred_keypoints") else np.zeros((0, 5, 3), dtype=float)
    scores = instances.scores.numpy() if instances.has("scores") else np.zeros((0,), dtype=float)
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros((0,), dtype=int)

    boxes, masks, keypoints = scale_prediction_data(boxes, masks, keypoints, image.shape, image_model.shape)
    classes_list, scores_list, boxes_list, masks_list, keypoints_list = filter_and_sort_detections(
        classes, scores, boxes, masks, keypoints, score_threshold=0.8
    )

    results = compute_cobb_results(classes_list, boxes_list, masks_list, keypoints_list, scores_list)
    results["input_shape"] = [int(image.shape[0]), int(image.shape[1])]
    results["model_shape"] = [int(image_model.shape[0]), int(image_model.shape[1])]
    results["detection_count"] = len(classes_list)
    return results


@app.post("/predict")
def predict_api(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = file.file.read()
    try:
        result = run_inference(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(content=result)


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    load_predictor()
    uvicorn.run(app, host="0.0.0.0", port=4000)
