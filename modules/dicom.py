# DICOM files: recognising them, and turning their pixel data into an 8-bit
# image the way a viewer shows it -- the file's own display window, with
# MONOCHROME1 inverted. Plain numpy and pydicom, no Qt, so it can be tested
# anywhere. modules/image_io.py uses this for display, upload and projects.

import math
import os

import numpy as np

from config import DICOM_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS

# Without a display window in the file, this share of the darkest and
# brightest pixels is clipped, so a few extreme pixels don't flatten the
# contrast.
AUTO_WINDOW_PERCENTILES = (0.5, 99.5)


class DicomReadError(Exception):
    """A DICOM file can't be read. The message is shown to the user."""


def is_dicom_file(path):
    """True for .dcm / .dicom files, and for other files that start like a
    DICOM file. PACS exports often have no extension, or a numeric one."""
    ext = os.path.splitext(path)[1].lower()
    if ext in DICOM_EXTENSIONS:
        return True
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return False
    try:
        with open(path, "rb") as f:
            f.seek(128)                     # DICOM files: 128-byte preamble, then "DICM"
            return f.read(4) == b"DICM"
    except OSError:
        return False


def read_dicom(path):
    """The image in a DICOM file as uint8 (see dicom_to_array). Raises
    DicomReadError with a readable message."""
    try:
        import pydicom
    except ImportError as exc:
        raise DicomReadError(
            "Reading DICOM files needs the pydicom package: "
            "python -m pip install -r requirements.txt"
        ) from exc
    try:
        return dicom_to_array(pydicom.dcmread(path))
    except Exception as exc:
        raise DicomReadError(f"Could not read this DICOM file: {exc}") from exc


def dicom_to_array(ds):
    """Pixel data of a pydicom dataset as uint8: H x W for grey, H x W x 3 for
    colour. Multi-frame files give their first frame."""
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    pixels = ds.pixel_array
    if int(ds.get("NumberOfFrames", 1) or 1) > 1:
        pixels = pixels[0]

    if int(ds.get("SamplesPerPixel", 1)) == 3:          # colour: already RGB
        if pixels.dtype == np.uint8:
            return np.ascontiguousarray(pixels)
        return scale_to_uint8(pixels, pixels.min(), pixels.max())

    values = apply_modality_lut(pixels, ds)
    if "VOILUTSequence" in ds:
        # A lookup table from the file: its output range is 0 .. 2^bits - 1.
        bits = int(ds.VOILUTSequence[0].LUTDescriptor[2])
        image = scale_to_uint8(apply_voi_lut(values, ds), 0, 2 ** bits - 1)
    elif "WindowCenter" in ds and "WindowWidth" in ds:
        image = apply_window(values, _first(ds.WindowCenter), _first(ds.WindowWidth),
                             str(ds.get("VOILUTFunction", "LINEAR")).upper())
    else:
        low, high = np.percentile(values, AUTO_WINDOW_PERCENTILES)
        image = scale_to_uint8(values, low, high)

    if ds.get("PhotometricInterpretation") == "MONOCHROME1":   # white = low values
        image = 255 - image
    return image


def apply_window(values, center, width, function="LINEAR"):
    """The DICOM display window (PS3.3 C.11.2.1.2) mapped to 0-255."""
    values = np.asarray(values, dtype=np.float64)
    width = max(float(width), 1.0)
    if function == "SIGMOID":
        scaled = 1.0 / (1.0 + np.exp(-4.0 * (values - center) / width))
    elif function == "LINEAR_EXACT":
        scaled = (values - center) / width + 0.5
    else:
        scaled = (values - (center - 0.5)) / max(width - 1.0, 1.0) + 0.5
    return np.round(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def scale_to_uint8(values, low, high):
    """Map low..high to 0..255, clipping outside it."""
    span = float(high) - float(low)
    if not math.isfinite(span) or span <= 0:
        return np.zeros(np.shape(values), dtype=np.uint8)
    scaled = (np.asarray(values, dtype=np.float64) - float(low)) / span
    return np.round(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def _first(value):
    """Window center/width can hold several values; the first is the default."""
    try:
        return float(value[0])
    except (TypeError, IndexError):
        return float(value)
