# Backend AI-model inference client.
#
# Talks to the Cobb Angle Inference API -- the same contract exercised in
# test_api_visualization.ipynb: POST an image file to a `/predict` endpoint,
# get back a JSON payload of detections/keypoints/angle_pairs. This module
# owns only the HTTP call and its error handling; parsing the JSON into
# clinical metrics is still ScoliosisModelEngine's job (modules/model_mock.py),
# and drawing it is OverlayLayer's job (modules/overlay.py).

import os
import time

import requests
from PySide6.QtCore import QThread, Signal

from config import INFERENCE_API_URL, INFERENCE_TIMEOUT

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _mime_for(path):
    return _MIME_TYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream")


class BackendUnavailableError(Exception):
    """Raised when the inference API can't be reached, times out, or errors."""


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

    elapsed = time.monotonic() - started
    return data, elapsed


class InferenceWorker(QThread):
    """Runs run_inference() off the UI thread so the app stays responsive
    while the model call is in flight (network + model inference time is
    not guaranteed to be instant)."""

    succeeded = Signal(dict, float)  # (result_dict, elapsed_seconds)
    failed = Signal(str)             # human-readable error message

    def __init__(self, image_path, api_url=INFERENCE_API_URL, timeout=INFERENCE_TIMEOUT, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.api_url = api_url
        self.timeout = timeout

    def run(self):
        try:
            data, elapsed = run_inference(self.image_path, self.api_url, self.timeout)
        except BackendUnavailableError as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(data, elapsed)
