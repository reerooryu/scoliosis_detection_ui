# Gets the AI result for one image, in one of two ways (Settings):
#   Local  -- runs the model on this computer (modules/inference.py);
#   Server -- POSTs the image to an inference server's /predict.
# Both give the same JSON result: detections, keypoints and angle pairs.
# This module only gets the result and turns failures into readable
# messages. ScoliosisModelEngine does the math and OverlayLayer the drawing.

import json
import logging
import os
import threading
import time

import requests
from PySide6.QtCore import QObject, Signal, Slot

from config import INFERENCE_API_URL, INFERENCE_TIMEOUT
from modules.app_paths import LOG_PATH
from modules.image_io import ImageLoadError, encoded_image
from modules.settings_dialog import MODE_LOCAL


logger = logging.getLogger(__name__)

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


class BackendUnavailableError(Exception):
    """Raised when the AI result can't be had: the server can't be reached,
    times out or errors, or the local model is missing or fails."""


def validate_inference_payload(data):
    """Check a result has the fields ScoliosisModelEngine relies on. Raises
    ValueError if not. Checking up front means bad data fails here with a
    clear message, not later in the middle of a landmark drag."""
    if not isinstance(data, dict):
        raise ValueError("response root is not an object")

    input_shape = data.get("input_shape")
    if (
        not isinstance(input_shape, list)
        or len(input_shape) != 2
        or any(not isinstance(value, (int, float)) or value <= 0 for value in input_shape)
    ):
        raise ValueError("input_shape must contain positive height and width values")

    detections = data.get("detections")
    if not isinstance(detections, list):
        raise ValueError("detections is not a list")
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            raise ValueError(f"detection {index} is not an object")
        keypoints = detection.get("keypoints")
        if not isinstance(keypoints, list) or len(keypoints) < 5:
            raise ValueError(f"detection {index} has fewer than five keypoints")
        for keypoint_index, keypoint in enumerate(keypoints):
            if (
                not isinstance(keypoint, (list, tuple))
                or len(keypoint) < 2
                or any(not isinstance(value, (int, float)) for value in keypoint[:2])
            ):
                raise ValueError(
                    f"detection {index} keypoint {keypoint_index} is invalid"
                )

    angle_pairs = data.get("angle_pairs")
    if not isinstance(angle_pairs, list):
        raise ValueError("angle_pairs is not a list")
    for index, pair in enumerate(angle_pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"angle pair {index} is not an object")
        for field in ("upper_detection_index", "lower_detection_index"):
            if not isinstance(pair.get(field), int):
                raise ValueError(f"angle pair {index} has an invalid {field}")

    return data


def _read_image(image_path):
    """The image file's bytes and extension. A DICOM file gives a PNG of its
    converted image (modules/image_io.py)."""
    try:
        return encoded_image(image_path)
    except (OSError, ImageLoadError) as exc:
        # The image was moved or deleted after it was loaded, or can't be read.
        raise BackendUnavailableError(f"Could not read the image file: {exc}") from exc


def run_inference(image_path, api_url=INFERENCE_API_URL, timeout=INFERENCE_TIMEOUT):
    """Server mode: POST the image at image_path to the server. Returns
    (result_dict, elapsed_seconds), the full round trip.

    The upload keeps the file's own name (often the hospital number), which
    helps when matching requests to cases.

    Raises BackendUnavailableError with a readable message on any failure,
    so the caller can let the user keep viewing the image and retry.
    """
    started = time.monotonic()
    image_bytes, ext = _read_image(image_path)

    try:
        files = {"file": (os.path.basename(image_path), image_bytes, _MIME_TYPES.get(ext, "application/octet-stream"))}
        response = requests.post(api_url, files=files, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise BackendUnavailableError(
            f"Could not reach the inference API at {api_url}.\n"
            "Is the backend model service running?"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise BackendUnavailableError(
            f"The inference API at {api_url} did not respond within {timeout}s."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise BackendUnavailableError(f"Inference request failed: {exc}") from exc

    if response.status_code != 200:
        raise BackendUnavailableError(
            f"Inference API returned HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise BackendUnavailableError("Inference API returned a response that wasn't valid JSON.") from exc

    try:
        validate_inference_payload(data)
    except ValueError as exc:
        raise BackendUnavailableError("Inference API returned an invalid result payload.") from exc

    elapsed = time.monotonic() - started
    return data, elapsed


def _import_inference():
    """modules.inference, imported on first use: it loads PyTorch, which
    takes a few seconds and is not needed in server mode."""
    try:
        from modules import inference
    except ImportError as exc:
        raise BackendUnavailableError(
            "Local mode needs the AI libraries (PyTorch, Detectron2), which are not "
            f"installed here ({exc}).\nInstall requirements-server.txt, or switch "
            "Settings → AI model to a server."
        ) from exc
    return inference


# Model files whose model is built and ready. Kept here rather than asking
# modules.inference, which may still be importing on another thread.
_loaded_models = set()


def is_local_model_loaded(weights_path):
    """True once the local model for weights_path is ready to use."""
    return os.path.abspath(weights_path or "") in _loaded_models


def load_local_model(weights_path):
    """Build the local model for weights_path (once; later calls return at
    once). Takes several seconds the first time."""
    if not weights_path:
        raise BackendUnavailableError(
            "No AI model file is chosen. Choose it in Settings → AI model."
        )
    if not os.path.isfile(weights_path):
        raise BackendUnavailableError(
            f"The AI model file was not found:\n{weights_path}\n"
            "Choose it again in Settings → AI model."
        )
    inference = _import_inference()
    try:
        predictor = inference.load_predictor(weights_path)
    except Exception as exc:
        raise BackendUnavailableError(f"The AI model file could not be loaded: {exc}") from exc
    _loaded_models.add(os.path.abspath(weights_path))
    return predictor


def run_local_inference(image_path, weights_path, on_loading=None):
    """Local mode: run the model on this computer. Returns
    (result_dict, elapsed_seconds). The time leaves out the one-time model
    load; on_loading() is called before that load, so the caller can say
    why the first analysis takes longer.

    Raises BackendUnavailableError with a readable message on any failure.
    """
    image_bytes, _ext = _read_image(image_path)
    if on_loading is not None and not is_local_model_loaded(weights_path):
        on_loading()
    predictor = load_local_model(weights_path)
    inference = _import_inference()

    started = time.monotonic()
    try:
        result = inference.run_inference(image_bytes, predictor)
    except Exception as exc:
        logger.exception("Local inference failed")
        raise BackendUnavailableError(f"The AI model could not analyse this image: {exc}") from exc
    elapsed = time.monotonic() - started

    # The same JSON round trip as a server reply, so both modes give plain
    # Python numbers and lists, never numpy types.
    data = json.loads(json.dumps(result))
    try:
        validate_inference_payload(data)
    except ValueError as exc:
        raise BackendUnavailableError("The AI model returned an invalid result.") from exc
    return data, elapsed


class InferenceWorker(QObject):
    """Runs one blocking inference request on a QThread that the caller
    creates and owns. settings is an InferenceSettings (modules/
    settings_dialog.py). A running request cannot be aborted, so cancel()
    only stops its result from being delivered."""

    succeeded = Signal(int, dict, float)  # (request_id, result, elapsed_seconds)
    failed = Signal(int, str)             # (request_id, human-readable message)
    status = Signal(int, str)             # (request_id, progress text for the status bar)
    finished = Signal(int)                # always emitted exactly once

    def __init__(self, request_id, image_path, settings, timeout=INFERENCE_TIMEOUT):
        super().__init__()
        self.request_id = request_id
        self.image_path = image_path
        self.settings = settings
        self.timeout = timeout
        self._cancelled = threading.Event()

    def cancel(self):
        """Mark this request as stale. Safe to call from any thread."""
        self._cancelled.set()

    @Slot()
    def run(self):
        if self._cancelled.is_set():
            self.finished.emit(self.request_id)
            return
        try:
            data, elapsed = self._get_result()
        except BackendUnavailableError as exc:
            if not self._cancelled.is_set():
                logger.warning("Inference request %s failed: %s", self.request_id, exc)
                self.failed.emit(self.request_id, str(exc))
        except Exception:
            # Report unexpected errors too, so the UI can offer a retry.
            if not self._cancelled.is_set():
                logger.exception("Unexpected error in inference request %s", self.request_id)
                self.failed.emit(
                    self.request_id,
                    "Inference could not be completed due to an unexpected error. "
                    f"Details are in the log file:\n{LOG_PATH}",
                )
        else:
            if not self._cancelled.is_set():
                self.succeeded.emit(self.request_id, data, elapsed)
        finally:
            self.finished.emit(self.request_id)

    def _get_result(self):
        if self.settings.mode == MODE_LOCAL:
            return run_local_inference(
                self.image_path, self.settings.weights_path, on_loading=self._on_loading
            )
        return run_inference(self.image_path, self.settings.api_url, self.timeout)

    def _on_loading(self):
        if not self._cancelled.is_set():
            self.status.emit(
                self.request_id, "Loading the AI model (first time only, may take up to a minute)…"
            )
