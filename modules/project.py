# Project files: a ".sdproj" lets a clinician save an assessment, including
# manual edits, and reopen it later without re-running the AI.
#
# It is a zip archive containing:
#   project.json        schema version, timestamp, original file name
#   image.<ext>         a copy of the X-ray, so the project still opens if
#                       the original file is moved or deleted
#   model_data.json     the current (possibly edited) result
#   baseline_data.json  the AI's original result, for "Reset Edits"

import json
import os
import zipfile
import zlib
from datetime import datetime, timezone

PROJECT_EXTENSION = ".sdproj"
SCHEMA_VERSION = 1

_METADATA_ENTRY = "project.json"
_MODEL_DATA_ENTRY = "model_data.json"
_BASELINE_DATA_ENTRY = "baseline_data.json"


class ProjectLoadError(Exception):
    """A .sdproj file is missing, not a zip, or corrupt. The message is
    written to be shown to the user."""


def write_project(model_engine, image_bytes, image_ext, original_filename, project_path):
    """Write the current result, the AI baseline and a copy of the image to
    project_path.

    Raises ValueError if no AI result is loaded, or OSError if the file
    cannot be written.
    """
    if not model_engine.has_baseline():
        raise ValueError("Cannot save a project before an AI result has been loaded.")

    parent_dir = os.path.dirname(project_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "original_filename": original_filename,
        "modified_at": datetime.now(timezone.utc).isoformat(),
    }
    image_ext = image_ext if image_ext.startswith(".") else f".{image_ext}"

    # Write to a temp file, then swap it in, so a failed save never leaves a
    # half-written project behind.
    tmp_path = project_path + ".tmp"
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(_METADATA_ENTRY, json.dumps(metadata, indent=2))
            zf.writestr(f"image{image_ext}", image_bytes)
            zf.writestr(_MODEL_DATA_ENTRY, json.dumps(model_engine.get_raw_data(), indent=2))
            zf.writestr(_BASELINE_DATA_ENTRY, json.dumps(model_engine.get_baseline_data(), indent=2))
        os.replace(tmp_path, project_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_project(project_path):
    """Read a .sdproj. Returns
    (image_bytes, image_ext, model_data, baseline_data, metadata).

    Raises ProjectLoadError with a readable message if the file is not valid.
    """
    if not os.path.exists(project_path):
        raise ProjectLoadError(f"Project file not found: {project_path}")

    try:
        with zipfile.ZipFile(project_path, "r") as zf:
            names = set(zf.namelist())

            if _METADATA_ENTRY not in names:
                raise ProjectLoadError("Not a valid project file (missing project.json).")
            metadata = json.loads(zf.read(_METADATA_ENTRY))
            if not isinstance(metadata, dict):
                raise ProjectLoadError("Not a valid project file (unreadable project.json).")

            version = metadata.get("schema_version")
            if version != SCHEMA_VERSION:
                raise ProjectLoadError(
                    f"Unsupported project file version ({version!r}); "
                    f"expected {SCHEMA_VERSION}."
                )

            image_name = next((n for n in names if n.startswith("image.")), None)
            if image_name is None:
                raise ProjectLoadError("Not a valid project file (missing embedded image).")
            image_bytes = zf.read(image_name)
            image_ext = os.path.splitext(image_name)[1]

            if _MODEL_DATA_ENTRY not in names or _BASELINE_DATA_ENTRY not in names:
                raise ProjectLoadError("Not a valid project file (missing detection data).")
            model_data = json.loads(zf.read(_MODEL_DATA_ENTRY))
            baseline_data = json.loads(zf.read(_BASELINE_DATA_ENTRY))
    except zipfile.BadZipFile as exc:
        raise ProjectLoadError("Not a valid project file (not a recognized archive).") from exc
    except (ValueError, KeyError, zlib.error) as exc:
        raise ProjectLoadError(f"Project file is corrupt or unreadable: {exc}") from exc
    except OSError as exc:
        raise ProjectLoadError(f"Could not read the project file: {exc}") from exc

    return image_bytes, image_ext, model_data, baseline_data, metadata
