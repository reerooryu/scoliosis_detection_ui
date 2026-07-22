# AGENTS.md — Scoliosis Detection & Measurement UI

This file orients a contributor working in this
repo. It covers the architecture, the live data flow, every module's job,
known dead code, and the workflow/gotchas that have bitten previous work
here. Read this before making changes — several bugs fixed in this repo's
history came from not knowing one of the things below.

## What this app is

A PySide6 (Qt6) desktop client for clinicians to load a spine X-ray, send it
to an AI inference backend (`server.py`, a FastAPI + Detectron2 service) that
detects vertebrae and predicts keypoints, then view/adjust the resulting
Cobb-angle measurements and export them as JSON.

Two independent runtimes live in this one repo:

1. **The desktop client** (`app.py` + `modules/`) — PySide6, runs on a
   clinician's machine, talks to the backend over HTTP. Dependencies:
   `requirements.txt`.
2. **The inference backend** (`server.py`) — FastAPI + Detectron2 +
   EfficientNet-B5 (via `timm`), a GPU-oriented model-serving process.
   Dependencies: `requirements-server.txt` (torch/detectron2 must be
   installed separately — see that file's comments; they're not normal
   pinnable PyPI packages).

These two normally run on **different machines** in a real deployment. They
are coupled by:
- The JSON contract described below (`test_api_visualization.ipynb` was the
  original reference for this contract).
- **`modules/geometry.py` is imported directly by `server.py`** for
  `cobb_angle_between_obliques` — this is the one place the backend reaches
  into client code. Keep this in mind if you ever try to run the backend
  somewhere `modules/` isn't deployed alongside it, or if you refactor
  `geometry.py`'s public function signatures.

## Running it

Client:
```
pip install -r requirements.txt
python app.py
```

Backend (needs a GPU box in practice, plus the model weights file that is
**not in this repo** — see "Confidential data" below):
```
pip install -r requirements-server.txt
# + torch and detectron2, installed separately to match your CUDA/OS (see
# comments in requirements-server.txt)
python server.py     # binds 0.0.0.0:4000, blocks on load_predictor() at startup
```

The client works fine with no backend running — it just shows the image and
lets the user retry (View → Retry AI Analysis, F5) once the backend is up.
The Settings dialog's "Inference API URL" (default
`http://127.0.0.1:4000/predict`, see `config.py`) points the client at it.

## Live architecture (the actual running app)

`app.py` → `modules/main_window.py:MainWindow` is the entire app. There is
**no wizard, no multi-page flow** — it's a single window with a
`QStackedWidget` toggling between two pages:

- **`modules/load_view.py:LoadPage`** — drag-and-drop / browse to pick an
  image, shows a thumbnail + Submit button. Knows nothing about the
  backend; just emits `submitted(image_path)`.
- **`modules/main_window.py:WorkspacePage`** — canvas + right-hand
  measurement panel (Cobb angles, apex, CSVL deviation, vertebra count,
  processing time) + Export/Reset buttons.

On Submit, `MainWindow._on_submit` shows the workspace immediately with just
the raw image, then kicks off `_run_inference`, which runs
`modules/parser.py:InferenceWorker` on a background `QThread` (never blocks
the UI). On success, the JSON result is handed to
`modules/model_mock.py:ScoliosisModelEngine`, which computes/recomputes all
clinical metrics, and `modules/overlay.py:OverlayLayer` draws everything on
the canvas. On failure, the image stays visible and the user can retry.

### Module map

| Module | Responsibility |
|---|---|
| `app.py` | Entry point. Creates `QApplication` + `MainWindow`. |
| `config.py` | All constants: window size, API URL/timeout, keypoint index names, overlay colors, handle radii. |
| `modules/main_window.py` | `MainWindow` (menu bar, toolbar, page-switching, inference lifecycle, undo/redo wiring, reset/settings/export/validation entry points) and `WorkspacePage` (canvas + measurement panel widget). This is the biggest, most central file. |
| `modules/load_view.py` | `LoadPage` / `DropZone` — image import UI only. No backend/model knowledge. |
| `modules/canvas.py` | `ImageCanvas(QGraphicsView)` — dumb base-image viewer (pan/zoom/fit). No AI/overlay knowledge. |
| `modules/overlay.py` | All `QGraphicsItem` subclasses for landmarks/outlines/Cobb lines/CSVL, plus `OverlayLayer`, which owns and incrementally updates them from a `ScoliosisModelEngine`. See "Overlay rendering" below — this file has the most subtle bugs fixed in it. |
| `modules/model_mock.py` | `ScoliosisModelEngine` — the clinical data/math state machine: loads a result (from disk or from the live API), scales coordinates to the displayed image, recalculates oblique/Cobb angles after any edit, and owns undo/redo + baseline/"Reset Edits". Despite the name ("mock"), this is used for **both** the demo JSON and live API results — the name is a holdover from before the real backend existed. |
| `modules/geometry.py` | Pure, unit-testable math: `oblique_angle` (exact port of `server.py`'s `cal_oblique01`), `cobb_angle_between_obliques` (shared with `server.py`), CSVL/apex computation, and the prediction-vs-label comparison functions used by `modules/validation.py`. |
| `modules/parser.py` | `run_inference()` (blocking HTTP POST to the backend) + `InferenceWorker(QObject)` (thread-safe wrapper run on a `QThread` by `MainWindow`). Validates the response shape (`_validate_inference_payload`) before it ever reaches UI code. |
| `modules/settings_dialog.py` | `SettingsDialog` — API URL, default Cobb-line color, default export folder, all persisted via `QSettings`. Also has a **fully commented-out** language switch (see "Known deferred work" below). |
| `modules/export.py` | `ExportDialog` — only "Raw JSON" is wired up (delegates to `modules/utils.py:export_json_data`, timestamped filename). Annotated Image / PDF / CSV are shown disabled ("coming soon"). |
| `modules/project.py` | `write_project()` / `read_project()` — save/reopen an in-progress assessment as a `.sdproj` zip bundle (image + current data + AI baseline), so a clinician can resume editing later without re-running inference. See "Project save/open" below. |
| `modules/validation.py` | `ValidationDialog` — a **separate, ML-team QA workflow** (Tools → Model Validation) comparing a prediction JSON against a ground-truth label JSON. Not part of the per-patient clinical flow; deliberately has no shared state with `MainWindow`. |
| `modules/utils.py` | `export_json_data`, `generate_clinical_summary` (plain-text clinical report string). |
| `modules/theme.py` | The dark clinical stylesheet + `apply_clinical_theme()`. Applied **only** to the toolbar/stack/status bar — see "Theming" below. |
| `server.py` | FastAPI backend: Detectron2 + custom EfficientNet-B5/FPN backbone, `/predict` and `/health` endpoints. See "Backend internals" below. |
| `modules/wizard.py`, `modules/pages.py` | **Dead code.** See "Known dead code" below — do not extend these. |

## The JSON contract (client ⟷ backend)

Both `server.py`'s `/predict` response and the bundled
`test_json/test_output.json` (used when running the client against no
backend — actually, note: `ScoliosisModelEngine.load_data()` reads this file,
but nothing in the live `MainWindow` flow calls it automatically anymore;
it's there for the engine's default-argument / manual testing path) follow
this shape:

```
{
  "input_shape": [height, width],
  "detection_count": int,
  "detections": [
    {
      "index": int, "class": int, "score": float | null,
      "box": [x1, y1, x2, y2],
      "upper_oblique": float, "lower_oblique": float,
      "keypoints": [[x,y,conf], ...]   # 5 points: center, TL, TR, BL, BR (see config.py KP_* constants)
    }, ...
  ],
  "angle_pairs": [
    {
      "pair_index": int,
      "upper_detection_index": int, "lower_detection_index": int,
      "upper_oblique": float, "lower_oblique": float,
      "cobb_angle": float
    }, ...
  ],
  "all_angles": [float, ...],
  "selected_cobb_angle": float | null,
  "upper_obliques": [float, ...], "lower_obliques": [float, ...]
}
```

`modules/parser.py:_validate_inference_payload` enforces the minimum shape
(`input_shape`, `detections[].keypoints` with ≥5 points, `angle_pairs[]`
index fields) at the network boundary, before anything reaches
`ScoliosisModelEngine` or the UI.

**Coordinate spaces**: the backend's `input_shape`/keypoints are in the
*original uploaded image's* pixel space. The client always calls
`ScoliosisModelEngine.scale_coordinates(displayed_width, displayed_height)`
after loading a result — in practice this is a no-op today since the
displayed pixmap is the same image at native resolution, but it's what makes
the coordinate math resolution-independent if that ever changes.

## Oblique / Cobb angle math — the one thing that must never drift

`server.py:cal_oblique01` and `modules/geometry.py:oblique_angle` **must
compute identically**. `oblique_angle` is a direct line-for-line port of
`cal_oblique01`, not a generic `atan2`. This matters because a plain
`math.atan2(dy, dx)` only agrees with this convention when the "left"
keypoint's x is less than the "right" keypoint's x (`dx >= 0`); whenever a
vertebra is rotated enough that this flips (`dx < 0` — exactly what happens
near a curve's apex), a plain atan2 diverges from the correct value by up to
180°. This was a real, previously-shipped bug. If you ever touch either
function, verify them against each other exhaustively across the dx/dy sign
combinations, not just a couple of manual test points.

Similarly, `modules/geometry.py:cobb_angle_between_obliques` is now
**imported directly by `server.py`** (not duplicated) — the client
recalculates Cobb angles locally after every keypoint drag using the exact
same function the backend used for its own initial result. Endplates are
undirected lines, so this function normalizes the periodic 180° ambiguity
before taking the acute angle between two obliques.

`ScoliosisModelEngine.recalculate_all_metrics()` **always overwrites** the
backend's own `upper_oblique`/`lower_oblique`/`cobb_angle` values with
locally-recomputed ones immediately after load (and after every edit) — so
in practice the client is the source of truth for displayed angles, and the
backend's own copies of those fields are only used for the very first paint
before the first recalculation.

## Overlay rendering — subtleties baked in from real bugs

`modules/overlay.py:OverlayLayer` is the most fragile-looking file here
because of several sharp edges already fixed once each:

- **Never destroy+recreate Cobb line/text scene items on every drag tick.**
  `_render_cobb_overlays` reuses existing `QGraphicsLineItem`/
  `QGraphicsTextItem`s via `setLine()`/`setHtml()` in place. A drag fires
  `keypoint_moved` on every mouse-move tick; destroying/recreating scene
  items from inside another item's `itemChange()` handler on every one of
  those ticks is a reentrant scene mutation that Qt Graphics View doesn't
  handle safely — this used to freeze/crash the app.
- **Never call `setPos()` on the handle currently being dragged.** `render()`
  skips `setPos()` for `handle.is_dragging == True` for the same reentrancy
  reason — Qt is already moving that exact item as part of the live drag.
- **`clear()` must tolerate already-deleted C++ objects.** `QGraphicsScene.
  clear()` (called by `ImageCanvas.load_image()`) deletes the underlying
  Qt/C++ item objects, but Python wrapper references can briefly still exist
  in `OverlayLayer`'s tracked lists. `_remove_item_if_valid` checks
  `shiboken6.isValid(item)` before touching anything — calling a Qt method on
  a dead wrapper raises `RuntimeError: Internal C++ object already deleted`.
- **Landmark handles and Cobb labels use `ItemIgnoresTransformations`.**
  Without it, ellipse radius / label font-size are scene units and scale
  with canvas zoom — handles became huge when zoomed in, and labels became
  tiny on a large image that's fit-to-view at a small scale. `HANDLE_RADIUS`
  / `ACTIVE_HANDLE_RADIUS` in `config.py` are screen pixels, not scene units.
- **Cobb label placement hugs each curve, not a fixed offset.**
  `_place_labels_without_overlap` computes, per curve, the scene-space
  horizontal extent of just the vertebrae that curve spans (not the whole
  spine — an S-shaped double curve can bulge in opposite directions at
  different heights), then places the label on whichever side (left/right)
  has more clear space in the *viewport*, hugging that edge by a small
  screen-pixel gap. This replaced two earlier, worse approaches: a fixed
  scene-unit offset (broke on wide/differently-framed images) and a global
  left-margin anchor (avoided anatomy but detached the label from its
  curve). Labels are nudged vertically per-side if they'd collide with the
  previous label on that same side, then clamped inside the viewport as a
  last-resort fallback.
- **Cobb line color is a live setting**, not a constant: `OverlayLayer` takes
  `cobb_line_color` in its constructor (from
  `SettingsDialog.get_saved_line_color()`) and `set_cobb_line_color()` is
  called when Settings is saved, updating existing `CobbMeasurementLineItem`s
  in place via `set_color()` (preserves line width).

## Undo/redo and "Reset Edits" vs. full "Reset"

`ScoliosisModelEngine` has two independent things that both sound like
"reset" — don't conflate them:

- **`capture_baseline()` / `has_edits()` / `reset_edits()`** — the AI's
  original, unedited result, captured **after** `scale_coordinates()` runs
  (order matters: capturing earlier would freeze the baseline in pre-scale
  coordinates). "Reset Edits" (Edit menu, Ctrl+Shift+R, toolbar button)
  restores this baseline without leaving the loaded image/analysis.
- **`snapshot_for_undo()` / `undo()` / `redo()`** — a capped (50-entry) deep-
  copy undo stack, pushed once per completed drag gesture (on
  `drag_started`, not per mouse-move tick), independent of the baseline.
- **`MainWindow._on_reset` / `WorkspacePage.reset_btn` / Edit → Full Reset
  (Ctrl+R)** — the actual "start over" action: clears the loaded image and
  analysis entirely and returns to `LoadPage`. This is deliberately **not**
  in the toolbar (only "Reset Edits" is) — it lives in the measurement
  panel below Export, and in the Edit menu.

Both `_on_reset` and `_on_reset_edits` (and `closeEvent`) guard against
silently discarding unexported changes via `_confirm_discard_if_dirty`,
gated on `MainWindow._dirty` (set on drag, cleared on successful export or
successful "Reset Edits").

## Project save/open

`File → Save Project` / `Save Project As...` / `Open Project...` (Ctrl+Shift+S /
no shortcut / Ctrl+Shift+O) let a clinician persist progress on one
in-progress assessment and resume it later, including manual landmark
edits, **without ever contacting the inference backend again**. This is
distinct from Export (`modules/export.py`): Export produces a one-way
clinical deliverable; a project is meant to be reopened and kept editing.

A `.sdproj` file (`modules/project.py`) is a zip bundle:
`project.json` (schema version + timestamp + original filename),
`image.<ext>` (an **embedded copy** of the source X-ray, not just a path
reference — so the project stays self-contained even if the original file
is later moved, renamed, deleted, or opened on a different machine),
`model_data.json` (`ScoliosisModelEngine.get_raw_data()` — the current,
possibly-edited detections), and `baseline_data.json` (the AI's original
result, via the new `get_baseline_data()` / `restore_baseline()` pair on
`ScoliosisModelEngine` — `restore_baseline()` sets `_original_data` from
saved data directly, unlike `capture_baseline()`, which always snapshots
whatever's currently loaded).

Three things to keep in mind if you touch this:
- **`AnalysisController.open_project()` deliberately does not render the
  overlay itself** — it loads the image into the canvas and returns; the
  actual `overlay_layer.render()` call lives in a separate
  `finish_open_project()`, invoked by `MainWindow._on_open_project()` via
  `QTimer.singleShot(150, ...)` *after* switching the stack to the
  workspace page. This mirrors `OverlayLayer`'s existing rule of never
  trusting the canvas's viewport size/transform until it's actually
  settled (see `ImageCanvas`'s own `showEvent`/`_delayed_initial_fit`, on
  the same 100ms-class timer): if the workspace page has never been shown
  in this run of the app, rendering the Cobb labels synchronously right
  after `open_project()` computes their screen positions against a
  stale/unsettled viewport — they land "way off" the image on that first
  open, then look correct on every subsequent open once the page's
  geometry has already settled. The Submit flow never hits this by
  accident, since the AI request's network round-trip already provides
  more than enough delay before its own `overlay_layer.render()` call.
- **`AnalysisSession` caches the source image bytes in memory**
  (`image_bytes`/`image_ext`/`original_filename`), not just `image_path`.
  A session from a fresh Submit only has a live file path until the first
  save — `AnalysisController.save_project()` reads and caches the bytes on
  that first save so every later save (and any save after the original
  file might move) no longer depends on that on-disk path at all. A
  session from `open_project()` has these populated immediately from the
  bundle, and `image_path` is `None` (there's no live file to retry
  inference against).
- **`session.dirty` (unexported edits) and `session.project_dirty`
  (unsaved-to-project changes) are independent flags**, both driven by the
  same edit operations (drag/undo/redo/reset edits) — you can export
  without saving a project, or save a project without exporting.
  `MainWindow._confirm_discard_if_dirty()` checks both together before
  Reset, Open Image, Open Project, or closing the window.

## Threading model (inference requests)

`MainWindow._run_inference` creates a fresh `QThread` + `InferenceWorker`
pair per request (never reuses one), tagged with an incrementing
`request_id`. Key points if you touch this:

- `requests` (the HTTP library) has no cross-thread abort API. `cancel()`
  just sets a `threading.Event`; an in-flight call still runs to its
  timeout, but its result is discarded (`_active_request_id` check in the
  `succeeded`/`failed` handlers) rather than updating stale UI — e.g. after
  the user hits Reset mid-request.
- Lifecycle teardown is the standard Qt chain:
  `worker.finished → thread.quit → thread.finished → {worker,thread}.
  deleteLater`. Both the `QThread` and `InferenceWorker` Python references
  are kept in `MainWindow._inference_jobs` until `thread.finished` fires —
  releasing them earlier risks premature C++ destruction.
- `closeEvent` refuses to close while any job is still outstanding.

## Backend internals (`server.py`)

- Custom Detectron2 backbone: `build_efficientnet_b5_fpn`, registered via
  `@BACKBONE_REGISTRY.register()`, wraps a `timm` EfficientNet-B5
  (`features_only=True`) in an FPN. Built once at first use.
- `load_predictor()` is idempotent and thread-safe via `_predictor_lock`
  (double-checked locking) — needed because FastAPI runs sync endpoints in a
  threadpool, so concurrent first requests could otherwise race to build
  the model twice.
- `/predict` is a **synchronous** (`def`, not `async def`) endpoint,
  deliberately — this lets FastAPI run it in its worker threadpool without
  blocking the async event loop (so `/health` stays responsive under load).
  Actual model inference is further serialized with `_inference_lock` since
  one shared `DefaultPredictor` isn't safe for concurrent GPU calls.
- `filter_and_sort_detections`: keeps only the **highest-confidence**
  class-1 detection if there are duplicates (not just the first one found),
  then sorts everything top-to-bottom by vertical box center.
- `compute_cobb_results` uses `cobb_angle_between_obliques` from
  `modules/geometry.py` (shared with the client — see the angle-math section
  above), and returns `score: None` when the caller didn't pass
  `scores_list` (kept optional for backward compatibility with earlier
  call sites).
- Startup requires the model weights file at
  `model/model_t001_6_effb5_mask_kp_2cls/model_final_run.pth`, which is
  **not in this repo** (see "Confidential data"). `load_predictor()` runs
  eagerly in `if __name__ == "__main__"` before `uvicorn.run`, so a missing
  weights file fails fast at startup rather than on the first request.

## Theming

`modules/theme.py`'s dark stylesheet is applied **only** to specific content
widgets (`MainWindow.toolbar`, `.stack`, `.statusBar()`) via
`apply_clinical_theme(*widgets)` — never to the `QApplication` or the
`QMainWindow` itself. This is intentional: the native menu bar and every
dialog (Settings, file/color pickers, message boxes, Export, Validation) stay
native OS chrome. If you add a new top-level dialog, don't theme it unless
that's a deliberate, discussed change.

Also note: plain `QWidget` subclasses embedded as non-top-level children
(`LoadPage`, `WorkspacePage`) need `setAttribute(Qt.WA_StyledBackground,
True)` or their stylesheet background silently doesn't paint — this bit
both of those widgets once already.

## Known dead code

`modules/wizard.py` (`ScoliosisWizard(QWizard)`) and `modules/pages.py`
(`ImageLoaderPage`, `InferencePage`, `AdjustmentPage`, `ExportPage`,
`DragDropArea`) are the **original multi-page wizard UI**, superseded by the
current single-window `MainWindow`. Nothing in the live `app.py → MainWindow`
import chain references them. `modules/pages.py` even references
`modules.canvas.ScoliosisInteractiveCanvas`, a class that **no longer exists**
in `modules/canvas.py` (today's `canvas.py` only has `ImageCanvas`) — so
these two files won't even import successfully as-is. Do not build on top of
them; they're kept around only for history/reference. If asked to clean up
the repo, these are safe to delete (confirm with the user first).

## Known deferred work (do not implement without asking)

- **Language switch (English/Thai)**: fully commented out in
  `modules/settings_dialog.py` (constants, combo box, save/load) at the
  user's explicit request, since Thai translations don't exist yet. Adding
  the switch back would only persist a choice — it would **not** retranslate
  any UI strings, since there's no i18n/string-table infrastructure in this
  app at all yet. See the comment block above `class SettingsDialog` for the
  exact 4 steps to re-enable once real translations exist.
- **Credits section**: the user asked for a reminder to add a credits
  section somewhere (likely Settings) before project completion — explicitly
  **not yet implemented**. Don't add it unprompted.
- **Export formats**: `modules/export.py` only implements Raw JSON. Annotated
  Image / PDF Report / CSV are shown as disabled "(coming soon)" checkboxes —
  they need format-specific renderers that don't exist yet (draw overlay
  onto image, lay out a PDF page, flatten keypoints to CSV rows).

## Confidential / not-in-repo data

Per `.gitignore`, the following are intentionally excluded from version
control (proprietary/large/in-house):
- `model/` — trained weights (`server.py`'s `WEIGHTS_PATH`), 500MB+ each.
- `test_api_visualization.ipynb` — the internal notebook the API contract was
  originally reverse-engineered from.
- `blueprint.md` — the original internal build-spec/prompt for this app.
- `detectron2_source/` — a vendored Detectron2 checkout used to run the
  server locally.

If you need to understand the "why" behind a clinical calculation and can't
find it in code comments, check whether `blueprint.md` or
`test_api_visualization.ipynb` exist locally (they won't be present in a
fresh clone) before guessing.

## Environment quirks specific to sandboxed/headless testing

Not part of the app itself, but relevant if you're verifying changes in a
headless Linux sandbox (no real display): PySide6/Qt needs
`QT_QPA_PLATFORM=offscreen` and the EGL/GLX/xkbcommon shared libraries
(`libEGL.so.1`, `libGLX.so.0`, `libxkbcommon.so.0`, `libOpenGL.so.0`,
`libGLdispatch.so.0`), which are not preinstalled — fetch them with
`apt-get download libegl1 libglx0 libxkbcommon0 libopengl0 libglvnd0` and
`dpkg-deb -x` each `.deb` into a local dir, then run with
`LD_LIBRARY_PATH` pointed at it. `/tmp` in that kind of sandbox can get wiped
between commands, which silently breaks this setup — if a previously-working
headless GUI test suddenly fails with a missing `.so`, re-extract rather than
assuming a code regression.

When editing files that keep failing `py_compile` right after a large
Edit/Write with no apparent reason (a `SyntaxError` mid-statement, cutting
off exactly at the tool-call boundary), suspect a mount-sync truncation
rather than an actual logic bug: rewrite the complete file via a shell
heredoc, then verify with **both** `py_compile` and an `ast.parse` pass that
explicitly checks for every expected function/class name (a truncated-but-
still-valid-Python file can pass plain `py_compile` while silently missing
trailing methods).
