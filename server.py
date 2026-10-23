# Inference server: runs the vertebra model on an uploaded X-ray and returns
# the result as JSON. Endpoints: POST /predict (an image file), GET /health.
#
# Steps for one image (run_inference):
#   1. decode the upload and resize it to the model input size;
#   2. run the model (Detectron2 keypoint R-CNN, EfficientNet-B5 backbone);
#   3. everything after that -- scaling back, filtering, angles, curves --
#      is plain math in modules/postprocess.py, testable without Detectron2.

from pathlib import Path
from threading import Lock
from typing import Dict

import cv2
import numpy as np
import timm
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from modules.postprocess import (
    SCORE_THRESHOLD, compute_cobb_results, filter_and_sort_detections, scale_prediction_data,
)

ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT_DIR / "model" / "model_t001_6_effb5_mask_kp_2cls" / "model_final_run.pth"

# Every image is resized to this before it goes into the model.
MODEL_INPUT_WIDTH = 768
MODEL_INPUT_HEIGHT = 1536

# Detectron2 drops boxes scoring at or below this before running the keypoint
# head, so the head only works on boxes that can pass the server's own filter
# (score >= SCORE_THRESHOLD). It must stay below SCORE_THRESHOLD: then the kept
# vertebrae and their keypoints are the same as with a lower value, because a
# box is only ever suppressed by a higher-scoring box of its class.
HEAD_SCORE_THRESHOLD = 0.75
assert HEAD_SCORE_THRESHOLD < SCORE_THRESHOLD

app = FastAPI(title="Cobb Angle Inference API")

predictor = None
# FastAPI runs sync endpoints in a threadpool, so the shared model is
# guarded: one lock to build it only once, one to run one inference at a time.
_predictor_lock = Lock()
_inference_lock = Lock()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TimmBackbone(Backbone):
    """A timm feature extractor exposed as a Detectron2 backbone."""

    def __init__(self, body):
        super().__init__()
        self.body = body
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = dict(zip(self._out_features, body.feature_info.channels()))

    def forward(self, x):
        return dict(zip(self._out_features, self.body(x)))

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_efficientnet_b5_fpn(cfg, input_shape: ShapeSpec):
    """Detectron2 backbone: timm EfficientNet-B5 features wrapped in an FPN."""
    body = timm.create_model("efficientnet_b5", pretrained=False, features_only=True, out_indices=(1, 2, 3, 4))
    bottom_up = TimmBackbone(body)
    return FPN(
        bottom_up=bottom_up,
        in_features=bottom_up._out_features,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm="",
        top_block=LastLevelMaxPool(),
    )


def build_config():
    """The model configuration the weights were trained with."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(WEIGHTS_PATH)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.DATASETS.TEST = ("vbs_test",)
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53]
    cfg.MODEL.PIXEL_STD = [58.395, 57.12, 57.375]
    cfg.INPUT.MIN_SIZE_TEST = MODEL_INPUT_WIDTH
    cfg.INPUT.MAX_SIZE_TEST = MODEL_INPUT_HEIGHT
    cfg.MODEL.BACKBONE.NAME = "build_efficientnet_b5_fpn"
    cfg.MODEL.FPN.OUT_CHANNELS = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.KEYPOINT_ON = True
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
    # The weights also contain a mask head (vertebra outlines). The result
    # does not use masks, and that head is by far the slowest part of the
    # model, so it is not built. It runs separately from the keypoint head,
    # so the keypoints are unchanged. Detectron2 logs that the mask weights
    # in the file are unused; that is expected.
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = HEAD_SCORE_THRESHOLD
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.45
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    return cfg


def load_predictor():
    """Build the model once and reuse it. Safe to call from several threads."""
    global predictor
    if predictor is not None:
        return predictor
    with _predictor_lock:
        # Another thread may have built it while this one waited for the lock.
        if predictor is None:
            predictor = DefaultPredictor(build_config())
    return predictor


# ---------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------

def read_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into an RGB image array."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_image_for_model(image: np.ndarray) -> np.ndarray:
    """Resize to the fixed model input size."""
    return cv2.resize(image, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)


def run_model(image_model: np.ndarray):
    """Run the model. Returns (boxes, keypoints, scores, classes) as numpy
    arrays, in model-input coordinates."""
    model = load_predictor()
    # One shared model, so only one inference runs at a time.
    with _inference_lock:
        outputs = model(image_model)
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        return np.zeros((0, 4)), np.zeros((0, 5, 3)), np.zeros((0,)), np.zeros((0,), dtype=int)
    return (
        instances.pred_boxes.tensor.numpy(),
        instances.pred_keypoints.numpy(),
        instances.scores.numpy(),
        instances.pred_classes.numpy(),
    )


def run_inference(image_bytes: bytes) -> Dict[str, object]:
    """Full pipeline for one image: decode, resize, predict, scale back to
    the original size, then compute the Cobb results."""
    image = read_image_bytes(image_bytes)
    image_model = prepare_image_for_model(image)
    boxes, keypoints, scores, classes = run_model(image_model)

    boxes, keypoints = scale_prediction_data(boxes, keypoints, image.shape, image_model.shape)
    classes_list, scores_list, boxes_list, keypoints_list = filter_and_sort_detections(
        classes, scores, boxes, keypoints
    )

    results = compute_cobb_results(classes_list, boxes_list, keypoints_list, scores_list)
    results["input_shape"] = [int(image.shape[0]), int(image.shape[1])]
    results["model_shape"] = [int(image_model.shape[0]), int(image_model.shape[1])]
    results["detection_count"] = len(classes_list)
    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

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
