from threading import Lock
from pathlib import Path
from typing import Dict

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

# Everything after the model runs (scaling, filtering, angles, curves) is
# plain math in modules/postprocess.py, testable without Detectron2.
from modules.postprocess import compute_cobb_results, filter_and_sort_detections, scale_prediction_data

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


def run_inference(image_bytes: bytes) -> Dict[str, object]:
    """Full pipeline for one image: decode, resize, predict, scale back to
    the original size, then compute the Cobb results."""
    image = read_image_bytes(image_bytes)
    image_model = prepare_image_for_model(image)
    predictor = load_predictor()
    # One shared model, so only one inference runs at a time.
    with _inference_lock:
        outputs = predictor(image_model)
    instances = outputs["instances"]
    # The masks are not used in the result. Drop them before copying to the
    # CPU: one full-size mask per detection, up to 300 of them, is by far the
    # largest part of the output.
    if instances.has("pred_masks"):
        instances.remove("pred_masks")
    instances = instances.to("cpu")

    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else np.zeros((0, 4), dtype=float)
    keypoints = instances.pred_keypoints.numpy() if instances.has("pred_keypoints") else np.zeros((0, 5, 3), dtype=float)
    scores = instances.scores.numpy() if instances.has("scores") else np.zeros((0,), dtype=float)
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros((0,), dtype=int)

    boxes, keypoints = scale_prediction_data(boxes, keypoints, image.shape, image_model.shape)
    classes_list, scores_list, boxes_list, keypoints_list = filter_and_sort_detections(
        classes, scores, boxes, keypoints, score_threshold=0.8
    )

    results = compute_cobb_results(classes_list, boxes_list, keypoints_list, scores_list)
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
