# Mock model inference and calculation engine for Scoliosis Detection UI

import copy
import json
import os
from config import DEFAULT_TEST_JSON_PATH, KP_TOP_LEFT, KP_TOP_RIGHT, KP_BOTTOM_LEFT, KP_BOTTOM_RIGHT
from modules.geometry import compute_csvl_x, compute_apex, oblique_angle

# How many completed drag edits to keep in the undo history. Each entry is a
# deep copy of the full detections payload, so this is capped to keep memory
# bounded for very long editing sessions.
UNDO_HISTORY_LIMIT = 50

class ScoliosisModelEngine:
    def __init__(self, json_path=DEFAULT_TEST_JSON_PATH, autoload=True):
        self.json_path = json_path
        self.data = None
        # Snapshot of the AI's original, un-edited result -- captured via
        # capture_baseline() once the data is fully loaded and scaled to the
        # displayed image. Lets "Reset Edits" discard manual adjustments
        # without losing the whole loaded image/analysis (that's the
        # toolbar's full Reset instead).
        self._original_data = None
        self._undo_stack = []
        self._redo_stack = []
        if autoload:
            self.load_data()

    def load_data(self):
        """Loads and parses the test_output.json file (demo/offline mode)."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Model mock data file not found at {self.json_path}")

        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

        # Ensure data consistency upon initial load
        self.recalculate_all_metrics()

    def load_from_dict(self, data):
        """Loads a result dict already fetched from the live inference API
        (see modules/parser.py) instead of reading it from disk. Used once
        the backend model is reachable; falls back to load_data() (the
        bundled test JSON) when it isn't."""
        self.data = data
        self.recalculate_all_metrics()

    def scale_coordinates(self, target_width, target_height):
        """
        Scales coordinates (bounding boxes and keypoints) to match the target image resolution.
        """
        input_shape = self.get_input_shape() # [height, width]
        orig_height, orig_width = input_shape[0], input_shape[1]

        if orig_width == target_width and orig_height == target_height:
            return  # No scaling needed

        scale_x = target_width / orig_width
        scale_y = target_height / orig_height

        # Scale each detection's box and keypoints
        for det in self.get_detections():
            if "box" in det and det["box"]:
                box = det["box"]
                det["box"] = [
                    box[0] * scale_x,
                    box[1] * scale_y,
                    box[2] * scale_x,
                    box[3] * scale_y
                ]

            if "keypoints" in det:
                for kp in det["keypoints"]:
                    kp[0] = kp[0] * scale_x
                    kp[1] = kp[1] * scale_y

        # Update input_shape to the new target dimensions
        self.data["input_shape"] = [target_height, target_width]

        # Recalculate metrics based on new coordinates
        self.recalculate_all_metrics()

    def get_detections(self):
        """Returns the list of vertebra detections."""
        return self.data.get("detections", [])

    def get_angle_pairs(self):
        """Returns the clinical Cobb angle pairs."""
        return self.data.get("angle_pairs", [])

    def get_selected_cobb_angle(self):
        """Returns the current main/selected Cobb angle."""
        return self.data.get("selected_cobb_angle", 0.0)

    def get_input_shape(self):
        """Returns original image input dimensions [height, width]."""
        return self.data.get("input_shape", [1832, 1190])

    def get_raw_data(self):
        """Returns the raw dictionary data."""
        return self.data

    def get_csvl_x(self):
        """X-coordinate of the CSVL (Central Sacral Vertical Line) reference
        -- see modules/geometry.py for how the reference vertebra is chosen."""
        return compute_csvl_x(self.get_detections())

    def get_apex(self):
        """Returns (apex_detection_index, deviation_px) -- the vertebra that
        deviates furthest from the CSVL."""
        return compute_apex(self.get_detections(), self.get_csvl_x())

    def update_keypoint(self, detection_index, keypoint_index, x, y):
        """
        Updates the coordinate of a specific keypoint of a vertebra.
        Triggers recalculation of oblique and Cobb angles.
        """
        detections = self.get_detections()
        if 0 <= detection_index < len(detections):
            det = detections[detection_index]
            keypoints = det.get("keypoints", [])
            if 0 <= keypoint_index < len(keypoints):
                # Update X, Y (keep confidence score unchanged)
                keypoints[keypoint_index][0] = float(x)
                keypoints[keypoint_index][1] = float(y)

                # Recalculate this vertebra's bounding box to encompass updated keypoints
                xs = [kp[0] for kp in keypoints]
                ys = [kp[1] for kp in keypoints]
                det["box"] = [min(xs), min(ys), max(xs), max(ys)]

                # Recalculate everything else
                self.recalculate_all_metrics()

    # ------------------------------------------------------------------
    # Baseline snapshot ("Reset Edits") and undo/redo
    # ------------------------------------------------------------------

    def capture_baseline(self):
        """Snapshots the current data as the AI's original result. Call this
        once, after the data is fully loaded AND scaled to the displayed
        image's resolution (main_window.py does this right after
        scale_coordinates()) -- capturing it any earlier would freeze in the
        pre-scale coordinates, which "Reset Edits" would then wrongly
        restore. Also clears undo/redo history, since both are meaningless
        once a new analysis has been loaded."""
        self._original_data = copy.deepcopy(self.data)
        self._undo_stack = []
        self._redo_stack = []

    def has_baseline(self):
        return self._original_data is not None

    def has_edits(self):
        """True if the current data differs from the captured baseline."""
        return self._original_data is not None and self.data != self._original_data

    def reset_edits(self):
        """Discards manual keypoint adjustments, restoring the AI's original
        detections (and clearing undo/redo history). Returns False if there
        is no baseline to restore (e.g. nothing has been loaded yet)."""
        if self._original_data is None:
            return False
        self.data = copy.deepcopy(self._original_data)
        self._undo_stack = []
        self._redo_stack = []
        self.recalculate_all_metrics()
        return True

    def snapshot_for_undo(self):
        """Pushes the current data onto the undo stack. Call this once per
        drag gesture, right as it *starts* (before any change is applied) --
        not per pixel of movement -- so each undo step corresponds to one
        completed adjustment, not to every intermediate mouse-move tick."""
        self._undo_stack.append(copy.deepcopy(self.data))
        if len(self._undo_stack) > UNDO_HISTORY_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack = []

    def can_undo(self):
        return len(self._undo_stack) > 0

    def can_redo(self):
        return len(self._redo_stack) > 0

    def undo(self):
        """Reverts the most recent completed drag. Returns False if there's
        nothing to undo."""
        if not self._undo_stack:
            return False
        self._redo_stack.append(copy.deepcopy(self.data))
        self.data = self._undo_stack.pop()
        self.recalculate_all_metrics()
        return True

    def redo(self):
        """Re-applies the most recently undone drag. Returns False if
        there's nothing to redo."""
        if not self._redo_stack:
            return False
        self._undo_stack.append(copy.deepcopy(self.data))
        self.data = self._redo_stack.pop()
        self.recalculate_all_metrics()
        return True

    def calculate_oblique_angle(self, p1, p2):
        """
        Calculates the oblique angle in degrees between two endplate corner
        points p1 (left) and p2 (right). Delegates to modules.geometry.oblique_angle,
        which is a direct port of server.py's cal_oblique01 -- NOT a plain
        atan2(dy, dx). This must match the backend's own convention exactly:
        a plain atan2 disagrees with cal_oblique01 by up to 180 degrees
        whenever a vertebra is rotated enough that its left/right keypoints'
        x-order flips (dx < 0) -- e.g. right at a curve's apex.
        """
        return oblique_angle(p1, p2)

    def recalculate_all_metrics(self):
        """
        Main mathematical recalculation engine.
        Updates:
          - Upper/lower obliques for all detections.
          - List of upper/lower obliques at the top level.
          - Cobb angles for all pre-defined angle pairs.
          - Maximum/selected Cobb angle.
        """
        detections = self.get_detections()

        # 1. Update oblique angles for each detection
        for det in detections:
            keypoints = det.get("keypoints", [])
            if len(keypoints) >= 5:
                # Top oblique (Keypoint 1 to Keypoint 2)
                p_tl = keypoints[KP_TOP_LEFT]
                p_tr = keypoints[KP_TOP_RIGHT]
                det["upper_oblique"] = self.calculate_oblique_angle(p_tl, p_tr)

                # Bottom oblique (Keypoint 3 to Keypoint 4)
                p_bl = keypoints[KP_BOTTOM_LEFT]
                p_br = keypoints[KP_BOTTOM_RIGHT]
                det["lower_oblique"] = self.calculate_oblique_angle(p_bl, p_br)

                # Recalculate center point (Keypoint 0) as average of corners
                p_center = keypoints[0]
                p_center[0] = sum(keypoints[i][0] for i in [1, 2, 3, 4]) / 4.0
                p_center[1] = sum(keypoints[i][1] for i in [1, 2, 3, 4]) / 4.0

        # 2. Update top-level obliques lists
        upper_obliques = [det["upper_oblique"] for det in detections]
        lower_obliques = [det["lower_oblique"] for det in detections]
        self.data["upper_obliques"] = upper_obliques
        self.data["lower_obliques"] = lower_obliques

        # 3. Recalculate Cobb angles for predefined pairs
        angle_pairs = self.get_angle_pairs()
        all_angles = []
        for pair in angle_pairs:
            u_idx = pair.get("upper_detection_index")
            l_idx = pair.get("lower_detection_index")

            if 0 <= u_idx < len(detections) and 0 <= l_idx < len(detections):
                u_det = detections[u_idx]
                l_det = detections[l_idx]

                # Get the relevant obliques
                u_oblique = u_det["upper_oblique"]
                l_oblique = l_det["lower_oblique"]

                # Calculate Cobb angle
                cobb = abs(u_oblique - l_oblique)
                pair["upper_oblique"] = u_oblique
                pair["lower_oblique"] = l_oblique
                pair["cobb_angle"] = cobb
                all_angles.append(cobb)

        self.data["all_angles"] = all_angles
        if all_angles:
            self.data["selected_cobb_angle"] = max(all_angles)
        else:
            self.data["selected_cobb_angle"] = 0.0

        # Update top level detection count
        self.data["detection_count"] = len(detections)
