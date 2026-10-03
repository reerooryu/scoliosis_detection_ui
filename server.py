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
from detectron2.engine import DefaultPredictor
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from modules.geometry import curve_start_index, select_angle_pairs

app = FastAPI(title="Cobb Angle Inference API")
ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT_DIR / "model" / "model_t001_6_effb5_mask_kp_2cls" / "model_final_run.pth"

predictor = None
# FastAPI runs sync endpoints in a threadpool, so the shared model is
# guarded: one lock to build it only once, one to run one inference at a time.
_predictor_lock = Lock()
_inference_lock = Lock()


@BACKBONE_REGISTRY.register()
def build_efficientnet_b5_fpn(cfg, input_shape: ShapeSpec):
    """Detectron2 backbone: timm EfficientNet-B5 features wrapped in an FPN."""
    body = timm.create_model(
        "efficientnet_b5",
        pretrained=False,
        features_only=True,
        out_indices=(1, 2, 3, 4),
    )

    # Run a dummy image through once to learn each feature map's channel count.
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
    """Build the model once and reuse it. Safe to call from several threads."""
    global predictor
    if predictor is not None:
        return predictor

    with _predictor_lock:
        # Another thread may have built it while this one waited for the lock.
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
    """Endplate tilt in degrees from its left corner (x1, y1) to its right
    corner (x2, y2). modules/geometry.py:oblique_angle is a copy of this and
    must give exactly the same results."""
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


def read_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into an RGB image array."""
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_image_for_model(image: np.ndarray) -> np.ndarray:
    """Resize to the fixed model input size: 768 wide by 1536 high."""
    return cv2.resize(image, (768, 1536), interpolation=cv2.INTER_LINEAR)


def scale_prediction_data(
    boxes: np.ndarray,
    masks: np.ndarray,
    keypoints: np.ndarray,
    original_shape: tuple,
    model_shape: tuple,
):
    """Map boxes, masks and keypoints from model-input size back to the
    original image size."""
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
    """Keep detections scoring at least score_threshold, keep only the best
    class-1 detection, and sort them top to bottom."""
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
        # Keep the most confident one, not just the first one found.
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

    results["upper_obliques"] = [float(x) for x in degree_upper]
    results["lower_obliques"] = [float(x) for x in degree_lower]

    # Curves: every curve above 10 degrees, each with the end vertebrae that
    # give its largest angle, searched from the marker vertebra downward.
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


def run_inference(image_bytes: bytes) -> Dict[str, object]:
    """Full pipeline for one image: decode, resize, predict, scale back to
    the original size, then compute the Cobb results."""
    image = read_image_bytes(image_bytes)
    image_model = prepare_image_for_model(image)
    predictor = load_predictor()
    # One shared model, so only one inference runs at a time.
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
    if (file.content_type or "").split("/")[0] != "image":
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

    # Load the model before opening the port, so a missing weights file
    # fails at startup instead of on the first request.
    load_predictor()
    uvicorn.run(app, host="0.0.0.0", port=4000)
