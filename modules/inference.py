# The AI model: build it, load the weights and run it on one image. Used by
# the server (server.py) and by the app's Local mode (modules/parser.py).
# Needs PyTorch, Detectron2, timm and OpenCV (requirements-server.txt).
#
# Steps for one image (run_inference):
#   1. decode the image bytes and resize them to the model input size;
#   2. run the model (Detectron2 keypoint R-CNN, EfficientNet-B5 backbone);
#   3. everything after that -- scaling back, filtering, angles, curves --
#      is plain math in modules/postprocess.py.

import os
from threading import Lock
from typing import Dict

import cv2
import numpy as np
import timm
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

from modules.app_paths import SOURCE_MODEL_PATH as DEFAULT_WEIGHTS_PATH
from modules.postprocess import (
    SCORE_THRESHOLD, compute_cobb_results, filter_and_sort_detections, scale_prediction_data,
)

# Every image is resized to this before it goes into the model.
MODEL_INPUT_WIDTH = 768
MODEL_INPUT_HEIGHT = 1536

# Detectron2 drops boxes scoring at or below this before running the keypoint
# head, so the head only works on boxes that can pass the final filter
# (score >= SCORE_THRESHOLD). It must stay below SCORE_THRESHOLD: then the kept
# vertebrae and their keypoints are the same as with a lower value, because a
# box is only ever suppressed by a higher-scoring box of its class.
HEAD_SCORE_THRESHOLD = 0.75
assert HEAD_SCORE_THRESHOLD < SCORE_THRESHOLD

# One model per weights file, built once and shared. Callers may be on
# several threads (FastAPI's threadpool, the app's worker threads): one lock
# to build each model only once, one to run one inference at a time.
_predictors = {}
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


def build_config(weights_path=DEFAULT_WEIGHTS_PATH):
    """The model configuration the weights were trained with. weights_path
    None builds the model with random weights (for self-tests)."""
    cfg = get_cfg()
    # The model was trained from Detectron2's "COCO-Keypoints/keypoint_rcnn_
    # R_50_FPN_3x.yaml". Its settings that matter for inference are set here
    # directly instead of reading that file, because reading it needs
    # pkg_resources and the file itself, both awkward in the packaged app.
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.5

    cfg.MODEL.WEIGHTS = str(weights_path) if weights_path else ""
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


def load_predictor(weights_path=DEFAULT_WEIGHTS_PATH):
    """The model for weights_path, built once and reused. Safe to call from
    several threads. Raises FileNotFoundError if the file is missing."""
    key = os.path.abspath(str(weights_path))
    predictor = _predictors.get(key)
    if predictor is not None:
        return predictor
    if not os.path.isfile(key):
        raise FileNotFoundError(f"Model file not found: {weights_path}")
    with _predictor_lock:
        # Another thread may have built it while this one waited for the lock.
        if key not in _predictors:
            _predictors[key] = DefaultPredictor(build_config(key))
    return _predictors[key]


def create_untrained_predictor():
    """A model with random weights, not cached. Only for self-tests that
    check the app can build and run the model (results are meaningless)."""
    return DefaultPredictor(build_config(None))


# ---------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------

def read_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode image file bytes into an RGB image array."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_image_for_model(image: np.ndarray) -> np.ndarray:
    """Resize to the fixed model input size."""
    return cv2.resize(image, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)


def run_model(predictor, image_model: np.ndarray):
    """Run the model. Returns (boxes, keypoints, scores, classes) as numpy
    arrays, in model-input coordinates."""
    # One shared model, so only one inference runs at a time.
    with _inference_lock:
        outputs = predictor(image_model)
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        return np.zeros((0, 4)), np.zeros((0, 5, 3)), np.zeros((0,)), np.zeros((0,), dtype=int)
    return (
        instances.pred_boxes.tensor.numpy(),
        instances.pred_keypoints.numpy(),
        instances.scores.numpy(),
        instances.pred_classes.numpy(),
    )


def run_inference(image_bytes: bytes, predictor) -> Dict[str, object]:
    """Full pipeline for one image file's bytes: decode, resize, predict,
    scale back to the original size, then compute the Cobb results."""
    image = read_image_bytes(image_bytes)
    image_model = prepare_image_for_model(image)
    boxes, keypoints, scores, classes = run_model(predictor, image_model)

    boxes, keypoints = scale_prediction_data(boxes, keypoints, image.shape, image_model.shape)
    classes_list, scores_list, boxes_list, keypoints_list = filter_and_sort_detections(
        classes, scores, boxes, keypoints
    )

    results = compute_cobb_results(classes_list, boxes_list, keypoints_list, scores_list)
    results["input_shape"] = [int(image.shape[0]), int(image.shape[1])]
    results["model_shape"] = [int(image_model.shape[0]), int(image_model.shape[1])]
    results["detection_count"] = len(classes_list)
    return results
