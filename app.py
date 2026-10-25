# Entry point for the desktop app.
#
#   python app.py                      start the app
#   python app.py --self-test REPORT   check the app can run the AI model, without
#                                      a window or model file; writes REPORT and
#                                      exits 0 (passed) or 1 (failed). Used by the
#                                      release build (.github/workflows/release.yml).
#   python app.py --self-test REPORT MODEL IMAGE
#                                      also run the real model file on an image and
#                                      save the result next to REPORT (REPORT.json).

import logging
import multiprocessing
import os
import sys
import traceback
from pathlib import Path

# Let `config` and `modules` import no matter where the app is launched from.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


from modules.app_paths import LOG_PATH  # noqa: E402  (needs sys.path above)


def setup_logging():
    """Log to LOG_PATH (replaced at each start). The packaged app has no
    console, so its print output and errors go there too; otherwise they are
    lost, and some libraries fail when there is nowhere to print."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(LOG_PATH, "w", encoding="utf-8", buffering=1)
    except OSError:
        log_file = open(os.devnull, "w")
    if getattr(sys, "frozen", False) or sys.stdout is None:
        sys.stdout = sys.stderr = log_file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(log_file)],
    )


def self_test(report_path, model_path=None, image_path=None):
    """Build the model with random weights and run it on a synthetic image,
    then check the Qt and DICOM parts load. With model_path and image_path,
    also run that model on that image. Returns the exit code."""
    lines, passed = [], True

    def step(name, check):
        nonlocal passed
        try:
            detail = check()
            lines.append(f"PASS {name}" + (f": {detail}" if detail else ""))
        except Exception:
            passed = False
            lines.append(f"FAIL {name}\n{traceback.format_exc()}")

    def check_qt():
        from PySide6 import __version__
        from PySide6.QtGui import QImage
        if QImage(8, 8, QImage.Format_RGB32).isNull():
            raise RuntimeError("Qt could not create an image")
        return f"PySide6 {__version__}"

    def check_dicom():
        import pydicom
        from pydicom.pixels.decoders import JPEGBaseline8BitDecoder, JPEG2000Decoder
        missing = [d.UID.name for d in (JPEGBaseline8BitDecoder, JPEG2000Decoder) if not d.is_available]
        if missing:
            raise RuntimeError("No decoder for: " + ", ".join(missing))
        return f"pydicom {pydicom.__version__}, JPEG and JPEG 2000 decoders available"

    def check_model():
        import time
        import cv2
        import numpy as np
        import torch
        from modules import inference
        from modules.parser import validate_inference_payload

        started = time.monotonic()
        predictor = inference.create_untrained_predictor()
        loaded = time.monotonic()
        # A grey gradient with bright bars, roughly the shape of a spine film.
        image = np.tile(np.linspace(30, 200, 600, dtype=np.uint8), (1200, 1))
        image[100:1100:60, 250:350] = 240
        ok, png = cv2.imencode(".png", image)
        assert ok
        result = inference.run_inference(png.tobytes(), predictor)
        validate_inference_payload(result)
        assert result["input_shape"] == [1200, 600], result["input_shape"]
        return (f"torch {torch.__version__}, device {predictor.cfg.MODEL.DEVICE}, build "
                f"{loaded - started:.1f}s, run {time.monotonic() - loaded:.1f}s, "
                f"{result['detection_count']} detections (random weights)")

    def check_model_file():
        import json
        import time
        from modules import inference
        from modules.parser import validate_inference_payload

        started = time.monotonic()
        predictor = inference.load_predictor(model_path)
        loaded = time.monotonic()
        with open(image_path, "rb") as image_file:
            result = inference.run_inference(image_file.read(), predictor)
        validate_inference_payload(result)
        Path(str(report_path) + ".json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        return (f"load {loaded - started:.1f}s, run {time.monotonic() - loaded:.1f}s, "
                f"{result['detection_count']} vertebrae, {len(result['angle_pairs'])} curves")

    step("Qt", check_qt)
    step("DICOM", check_dicom)
    step("AI model", check_model)
    if model_path and image_path:
        step("Model file on image", check_model_file)
    lines.append("PASSED" if passed else "FAILED")
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if passed else 1


def main():
    multiprocessing.freeze_support()
    setup_logging()

    if len(sys.argv) >= 3 and sys.argv[1] == "--self-test":
        sys.exit(self_test(*sys.argv[2:5]))

    from PySide6.QtWidgets import QApplication
    from modules.main_window import MainWindow

    app = QApplication(sys.argv)

    # MainWindow themes only its toolbar and workspace; menus and dialogs stay native.
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
