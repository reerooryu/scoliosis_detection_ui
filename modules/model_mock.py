# Mock model inference and calculation engine for Scoliosis Detection UI

import json
import os
import math
from config import DEFAULT_TEST_JSON_PATH, KP_TOP_LEFT, KP_TOP_RIGHT, KP_BOTTOM_LEFT, KP_BOTTOM_RIGHT

class ScoliosisModelEngine:
    def __init__(self, json_path=DEFAULT_TEST_JSON_PATH):
        self.json_path = json_path
        self.data = None
        self.load_data()

    def load_data(self):
        """Loads and parses the test_output.json file."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Model mock data file not found at {self.json_path}")
            
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
            
        # Ensure data consistency upon initial load
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

    def calculate_oblique_angle(self, p1, p2):
        """
        Calculates oblique angle in degrees between two points p1 (left) and p2 (right).
        Angle is atan2(y2 - y1, x2 - x1) converted to degrees.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle_rad = math.atan2(dy, dx)
        return math.degrees(angle_rad)

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
