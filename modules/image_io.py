# Image files: reading X-rays for display, upload and project files.
#
# JPG and PNG are used as they are. A DICOM file is converted to an 8-bit
# image first (modules/dicom.py), so the rest of the app, and the server,
# only ever see a normal image. DICOM header data (patient name, ID ...) is
# never passed on.

import os
from functools import lru_cache

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QImage

from config import SUPPORTED_IMAGE_EXTENSIONS
from modules.dicom import DicomReadError, is_dicom_file, read_dicom


class ImageLoadError(Exception):
    """The file can't be read as an X-ray. The message is shown to the user."""


def is_supported_image(path):
    return os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS or is_dicom_file(path)


def load_image(path):
    """The image for display, as a QImage. Raises ImageLoadError.

    The last couple of files are cached, so the Load page preview and the
    workspace don't decode a large DICOM twice. Call from the GUI thread."""
    try:
        stat = os.stat(path)
    except OSError as exc:
        raise ImageLoadError(f"Could not open the file: {exc}") from exc
    return _load_cached(path, stat.st_mtime_ns, stat.st_size)


def encoded_image(path):
    """The image as file bytes plus its extension, for upload or a project.
    JPG / PNG: the file itself, unchanged. DICOM: a lossless PNG of the
    converted image. Safe to call from a worker thread (it doesn't use the
    cache). Raises ImageLoadError or OSError."""
    if not is_dicom_file(path):
        with open(path, "rb") as f:
            return f.read(), os.path.splitext(path)[1].lower()
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.WriteOnly)
    if not _dicom_qimage(path).save(buffer, "PNG"):
        raise ImageLoadError("Could not convert the DICOM image.")
    return bytes(data), ".png"


@lru_cache(maxsize=2)
def _load_cached(path, _mtime_ns, _size):
    if is_dicom_file(path):
        return _dicom_qimage(path)
    image = QImage(path)
    if image.isNull():
        raise ImageLoadError("Invalid or corrupted image file.")
    return image


def _dicom_qimage(path):
    try:
        pixels = np.ascontiguousarray(read_dicom(path))
    except DicomReadError as exc:
        raise ImageLoadError(str(exc)) from exc
    height, width = pixels.shape[:2]
    fmt = QImage.Format_Grayscale8 if pixels.ndim == 2 else QImage.Format_RGB888
    # copy() so the QImage owns its pixels after the numpy array is freed.
    return QImage(pixels.data, width, height, pixels.strides[0], fmt).copy()
