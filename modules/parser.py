# Backend AI-model inference client.
#
# Talks to the Cobb Angle Inference API -- the same contract exercised in
# test_api_visualization.ipynb: POST an image file to a `/predict` endpoint,
# get back a JSON payload of detections/keypoints/angle_pairs. This module
# owns only the HTTP call and its error handling; parsing the JSON into
# clinical metrics is still ScoliosisModelEngine's job (modules/model_mock.py),
# and drawing it is OverlayLayer's job (modules/overlay.py).

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


def _validate_inference_payload(data):
    """Validate the minimum API contract consumed by ScoliosisModelEngine.

    Keeping this at the network boundary prevents malformed JSON from failing
    later in a UI slot or during a landmark drag, where it is much harder to
    report a useful recovery action.
    """
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
    """POSTs image_path to the inference API and returns (result_dict, elapsed_seconds).

    Raises BackendUnavailableError with a clinician-readable message on
    connection failure, timeout, or a non-200 response -- callers should
    catch this and let the user keep viewing/re-trying rather than crash.
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

    if response.status_code != 200:
        raise BackendUnavailableError(
            f"Inference API returned HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise BackendUnavailableError("Inference API returned a response that wasn't valid JSON.") from exc

    try:
        _validate_inference_payload(data)
    except ValueError as exc:
        raise BackendUnavailableError("Inference API returned an invalid result payload.") from exc

    elapsed = time.monotonic() - started
    return data, elapsed


class InferenceWorker(QObject):
    """Performs one blocking inference request in a dedicated QThread.

    This class deliberately owns no QThread.  The UI creates the thread,
    moves this worker into it, and tears both objects down through their Qt
    lifecycle signals.  ``requests`` cannot abort an in-flight upload/read;
    cancellation therefore suppresses delivery of an obsolete result as soon
    as the request returns.
    """

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
        """Thread-safe invalidation for a superseded request.

        The active requests call remains bounded by its configured timeout;
        its result is discarded rather than allowed to update the UI.
        """
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
            # Keep unexpected worker errors from silently terminating a
            # background thread without restoring the UI's retry state.
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
