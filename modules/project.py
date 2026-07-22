# Project save/load: a self-contained ".sdproj" bundle that lets a clinician
# save progress on one in-progress assessment and resume it later --
# including any manual landmark edits -- without re-running AI inference.
#
# A project file is a zip archive (the same trick .docx/.pptx use) so an
# image, JSON data, and metadata can sit side-by-side without inventing a
# custom binary format:
#
#   project.json        -- schema version + timestamp + original filename
#   image.<ext>          -- a copy of the source X-ray, embedded (not just a
#                            path reference -- so a project stays fully
#                            self-contained even if the original file is
#                            later moved, renamed, or deleted, or the
#                            project is opened on a different machine)
#   model_data.json      -- the current (possibly edited) detections /
#                            keypoints / angles, i.e. ScoliosisModelEngine's
#                            live data (get_raw_data())
#   baseline_data.json   -- the AI's original, unedited result, so "Reset
#                            Edits" still works correctly after reopening
#
# Opening a project never talks to the inference backend -- everything
# needed to redraw the canvas and measurement panel already lives in the
# file. modules/controller.py:AnalysisController is the only thing that
# calls into this module; see its save_project()/open_project().

import json
import os
import zipfile
from datetime import datetime, timezone

PROJECT_EXTENSION = ".sdproj"
SCHEMA_VERSION = 1

_METADATA_ENTRY = "project.json"
_MODEL_DATA_ENTRY = "model_data.json"
_BASELINE_DATA_ENTRY = "baseline_data.json"


class ProjectLoadError(Exception):
    """Raised when a .sdproj file is missing, not a zip, or has an
    unsupported/corrupt schema -- callers should catch this the same way
    modules/parser.py's BackendUnavailableError is handled (surfaced as a
    clinician-readable message, never a raw traceback)."""


def write_project(model_engine, image_bytes, image_ext, original_filename, project_path):
    """Writes model_engine's current + baseline data, plus a copy of the
    source image, to project_path (a .sdproj zip bundle).

    Raises ValueError if model_engine has no captured baseline yet (nothing
    resembling a completed assessment to save), or OSError if the file can't
    be written.
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

    # Write to a temporary path and replace atomically -- an interrupted
    # write (crash, full disk) should never leave a half-written zip sitting
    # at the destination the user picked, especially when that destination
    # is an existing project file being overwritten by a routine save.
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
    """Reads a .sdproj bundle back into
    (image_bytes, image_ext, model_data, baseline_data, metadata).

    Raises ProjectLoadError with a clinician-readable message if the file is
    missing, isn't a zip, or is missing/has mismatched required entries.
    """
    if not os.path.exists(project_path):
        raise ProjectLoadError(f"Project file not found: {project_path}")

    try:
        with zipfile.ZipFile(project_path, "r") as zf:
            names = set(zf.namelist())

            if _METADATA_ENTRY not in names:
                raise ProjectLoadError("Not a valid project file (missing project.json).")
            metadata = json.loads(zf.read(_METADATA_ENTRY))

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
    except (json.JSONDecodeError, KeyError) as exc:
        raise ProjectLoadError(f"Project file is corrupt or unreadable: {exc}") from exc

    return image_bytes, image_ext, model_data, baseline_data, metadata
