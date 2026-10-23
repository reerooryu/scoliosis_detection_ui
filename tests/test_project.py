# Save a .sdproj project and open it again: everything must come back
# unchanged, including manual edits and the AI's original result.

import os

import pytest

from modules.model_mock import ScoliosisModelEngine
from modules.project import PROJECT_EXTENSION, ProjectLoadError, read_project, write_project

from synthetic_spine import make_result

# Any bytes work: a project stores the image as-is, without decoding it.
IMAGE_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(256))


def edited_engine():
    """A loaded result with one landmark moved, so it differs from the AI's."""
    engine = ScoliosisModelEngine()
    engine.load_from_dict(make_result())
    engine.set_processing_time(3.2104)
    engine.capture_baseline()
    x, y = engine.get_detections()[10]["keypoints"][1][:2]
    engine.update_keypoint(10, 1, x + 12, y - 7)
    engine.reselect_curves()
    assert engine.has_edits()
    return engine


def save(engine, folder):
    path = os.path.join(str(folder), "case" + PROJECT_EXTENSION)
    write_project(engine, IMAGE_BYTES, ".png", "case.png", path)
    return path


def test_save_and_reopen_returns_the_same_data(tmp_path):
    engine = edited_engine()
    path = save(engine, tmp_path)

    image_bytes, image_ext, model_data, baseline_data, metadata = read_project(path)

    assert image_bytes == IMAGE_BYTES
    assert image_ext == ".png"
    assert metadata["original_filename"] == "case.png"
    assert model_data == engine.get_raw_data()
    assert baseline_data == engine.get_baseline_data()
    assert not os.path.exists(path + ".tmp")     # the temporary file is cleaned up


def test_reopened_project_gives_the_same_measurements(tmp_path):
    engine = edited_engine()
    path = save(engine, tmp_path)
    _, _, model_data, baseline_data, _ = read_project(path)

    reopened = ScoliosisModelEngine()
    reopened.load_from_dict(model_data)
    reopened.restore_baseline(baseline_data)

    assert reopened.get_angle_pairs() == engine.get_angle_pairs()
    assert reopened.get_selected_cobb_angle() == engine.get_selected_cobb_angle()
    assert reopened.get_processing_time() == 3.21
    assert reopened.has_edits()

    # "Reset Edits" after reopening returns to the AI's original result.
    reopened.reset_edits()
    assert reopened.get_raw_data() == engine.get_baseline_data()


def test_a_file_that_is_not_a_project_gives_a_clear_error(tmp_path):
    path = tmp_path / ("broken" + PROJECT_EXTENSION)
    path.write_bytes(b"not a zip file")
    with pytest.raises(ProjectLoadError):
        read_project(str(path))
