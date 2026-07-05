# Configuration and constants for the Scoliosis Detection UI application

import os

# Application Window Configuration
APP_NAME = "Scoliosis Detection & Measurement UI"
WINDOW_WIDTH = 950
WINDOW_HEIGHT = 950

# Data paths
DEFAULT_TEST_JSON_PATH = os.path.join("test_json", "test_output.json")

# Keypoint Indices Map
KP_CENTER = 0
KP_TOP_LEFT = 1
KP_TOP_RIGHT = 2
KP_BOTTOM_LEFT = 3
KP_BOTTOM_RIGHT = 4

# Visual Settings (Colors and Opacity)
# We use standard PySide6 colors (QColor)
COLOR_VERTEBRA_OUTLINE = (0, 120, 215, 180)     # Blue, semi-transparent
COLOR_CORRIDOR_FILL = (0, 204, 150, 60)         # Emerald Green, low opacity (alpha = 60/255)
COLOR_KEYPOINT_CENTER = (255, 193, 7, 255)      # Amber/Yellow for center
COLOR_KEYPOINT_CORNER = (220, 53, 69, 255)      # Red for corner points
COLOR_COBB_LINE = (255, 87, 34, 200)            # Deep Orange for Cobb angle lines
COLOR_COBB_TEXT_BG = (255, 255, 255, 220)       # White for text background

# Drag Handle Radius
HANDLE_RADIUS = 5.0
ACTIVE_HANDLE_RADIUS = 7.0

# Clinical Parameters
# Default Cobb Angle measurement pairs from JSON file
DEFAULT_ANGLE_PAIRS = [
    {"upper": 5, "lower": 11},
    {"upper": 11, "lower": 18}
]
