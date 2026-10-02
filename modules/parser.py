# Client for the inference server: POST an image to /predict and get back
# detections, keypoints and angle pairs as JSON. This module only makes the
# HTTP call and handles its errors. ScoliosisModelEngine does the math and
# OverlayLayer does the drawing.

import logging
import os
import threading
import time

import requests
from PySide6.QtCore import QObject, Signal, Slot

from config import INFERENCE_API_URL, INFERENCE_TIMEOUT


logger = logging.getLogger(__name__)

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _mime_for(path):
    return _MIME_TYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream")


class BackendUnavailableError(Exception):
    """Raised when the inference API can't be reached, times out, or errors."""


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


def run_inference(image_path, api_url=INFERENCE_API_URL, timeout=INFERENCE_TIMEOUT):
    """POST image_path to the server. Returns (result_dict, elapsed_seconds).

    Raises BackendUnavailableError with a readable message on any failure,
    so the caller can let the user keep viewing the image and retry.
    """
    started = time.monotonic()
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, _mime_for(image_path))}
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
    except OSError as exc:
        # The image was moved or deleted after it was loaded.
        raise BackendUnavailableError(f"Could not read the image file: {exc}") from exc

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


class InferenceWorker(QObject):
    """Runs one blocking inference request on a QThread that the caller
    creates and owns. A running request cannot be aborted, so cancel() only
    stops its result from being delivered."""

    succeeded = Signal(int, dict, float)  # (request_id, result, elapsed_seconds)
    failed = Signal(int, str)             # (request_id, human-readable message)
    finished = Signal(int)                # always emitted exactly once

    def __init__(self, request_id, image_path, api_url=INFERENCE_API_URL, timeout=INFERENCE_TIMEOUT):
        super().__init__()
        self.request_id = request_id
        self.image_path = image_path
        self.api_url = api_url
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
            data, elapsed = run_inference(self.image_path, self.api_url, self.timeout)
        except BackendUnavailableError as exc:
            if not self._cancelled.is_set():
                logger.warning("Inference request %s failed: %s", self.request_id, exc)
                self.failed.emit(self.request_id, str(exc))
        except Exception as exc:
            # Report unexpected errors too, so the UI can offer a retry.
            if not self._cancelled.is_set():
                logger.exception("Unexpected error in inference request %s", self.request_id)
                self.failed.emit(
                    self.request_id,
                    "Inference could not be completed due to an unexpected error. "
                    "Check the application logs for details.",
                )
        else:
            if not self._cancelled.is_set():
                self.succeeded.emit(self.request_id, data, elapsed)
        finally:
            self.finished.emit(self.request_id)
