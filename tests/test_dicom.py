# Tests for modules/dicom.py: recognising DICOM files and converting their
# pixel data to an 8-bit image. Needs pydicom and numpy, not Qt. The DICOM
# files are generated here: no patient data.

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, RLELossless, generate_uid

from modules.dicom import DicomReadError, apply_window, dicom_to_array, is_dicom_file, read_dicom


def make_dataset(pixels, photometric="MONOCHROME2", bits_stored=12, **attributes):
    """A minimal DX-like dataset holding pixels (2-D, or 3-D for frames/colour)."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"   # Digital X-Ray
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "DX"
    ds.set_pixel_data(pixels, photometric, bits_stored)
    for name, value in attributes.items():
        setattr(ds, name, value)
    return ds


def ramp(height=8, width=256, top=4095):
    return np.tile(np.linspace(0, top, width).round().astype(np.uint16), (height, 1))


# ---------------------------------------------------------------------------
# Display window
# ---------------------------------------------------------------------------

def test_window_maps_center_to_mid_grey_and_clips_outside():
    out = apply_window(np.array([0, 1000, 2000, 3000, 4000]), center=2000, width=2000)
    assert out[0] == 0 and out[1] == 0          # at or below the window's low edge
    assert abs(int(out[2]) - 128) <= 1          # the center
    assert out[3] == 255 and out[4] == 255      # at or above the high edge


def test_window_from_the_file_is_used():
    ds = make_dataset(ramp(), WindowCenter=2048, WindowWidth=1024)
    image = dicom_to_array(ds)
    assert image.dtype == np.uint8 and image.shape == (8, 256)
    assert image[0, 0] == 0 and image[0, -1] == 255     # outside the window: clipped
    assert np.all(np.diff(image[0].astype(int)) >= 0)   # brighter to the right


def test_rescale_is_applied_before_the_window():
    # Stored values 0, 16 ... 4080, rescaled by x2 + 100. A window centred on
    # 4196 (the rescaled value of stored 2048) puts stored 2048 at mid grey.
    pixels = np.tile(np.arange(0, 4096, 16, dtype=np.uint16), (8, 1))
    ds = make_dataset(pixels, RescaleSlope=2, RescaleIntercept=100, WindowCenter=4196, WindowWidth=400)
    image = dicom_to_array(ds)
    middle = int(np.where(pixels[0] == 2048)[0][0])
    assert abs(int(image[0, middle]) - 128) <= 1


def test_no_window_stretches_to_the_full_range():
    ds = make_dataset(ramp(top=1000) + 500)
    image = dicom_to_array(ds)
    assert image.min() == 0 and image.max() == 255


def test_voi_lut_table_is_used_when_present():
    lut = Dataset()
    lut.LUTDescriptor = [4096, 0, 16]                     # 4096 entries, from 0, 16-bit output
    lut.LUTData = list(np.linspace(65535, 0, 4096).round().astype(int))   # an inverting table
    ds = make_dataset(ramp(), VOILUTSequence=Sequence([lut]))
    image = dicom_to_array(ds)
    assert image[0, 0] == 255 and image[0, -1] == 0


def test_monochrome1_is_inverted():
    pixels = ramp()
    normal = dicom_to_array(make_dataset(pixels, "MONOCHROME2", WindowCenter=2048, WindowWidth=4096))
    inverted = dicom_to_array(make_dataset(pixels, "MONOCHROME1", WindowCenter=2048, WindowWidth=4096))
    assert np.array_equal(inverted, 255 - normal)


def test_multi_frame_gives_the_first_frame():
    frames = np.stack([ramp(), ramp()[:, ::-1]])
    ds = make_dataset(frames, WindowCenter=2048, WindowWidth=4096)
    ds.NumberOfFrames = 2
    assert np.array_equal(dicom_to_array(ds), dicom_to_array(make_dataset(ramp(), WindowCenter=2048, WindowWidth=4096)))


def test_colour_is_passed_through():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    image = dicom_to_array(make_dataset(rgb, "RGB", bits_stored=8))
    assert image.shape == (4, 5, 3) and np.array_equal(image, rgb)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def save(ds, path, compress=None):
    if compress:
        ds.compress(compress)
    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


def test_dicom_files_are_recognised_with_or_without_an_extension(tmp_path):
    for name in ("xray.dcm", "XRAY.DICOM", "IM0001", "1.2.840.113619.2.55"):
        assert is_dicom_file(save(make_dataset(ramp()), tmp_path / name)), name
    png = tmp_path / "xray.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert not is_dicom_file(str(png))
    other = tmp_path / "notes"
    other.write_bytes(b"not a dicom file at all" * 10)
    assert not is_dicom_file(str(other))


def test_reading_a_file_matches_the_dataset(tmp_path):
    ds = make_dataset(ramp(), WindowCenter=2048, WindowWidth=2048)
    expected = dicom_to_array(ds)
    assert np.array_equal(read_dicom(save(ds, tmp_path / "plain.dcm")), expected)


def test_compressed_files_are_read(tmp_path):
    ds = make_dataset(ramp(), WindowCenter=2048, WindowWidth=2048)
    expected = dicom_to_array(ds)
    path = save(make_dataset(ramp(), WindowCenter=2048, WindowWidth=2048), tmp_path / "rle.dcm", compress=RLELossless)
    assert np.array_equal(read_dicom(path), expected)


def test_a_broken_file_gives_a_readable_error(tmp_path):
    path = tmp_path / "broken.dcm"
    path.write_bytes(b"\0" * 128 + b"DICM" + b"garbage")
    with pytest.raises(DicomReadError):
        read_dicom(str(path))
