# Configuration and constants for the Scoliosis Detection UI application

import os

# Application Window Configuration
APP_NAME = "Scoliosis Detection & Measurement UI"
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900

# Image import validation (Open Image feature)
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Backend AI-model inference API (see test_api_visualization.ipynb for the
# reference contract this was captured against). Overridable per-install via
# the Settings dialog, persisted through QSettings.
INFERENCE_API_URL = "http://127.0.0.1:4000/predict"
INFERENCE_TIMEOUT = 120  # seconds

# Data paths
# Anchored to this file's directory (not the process cwd) so the app works
# regardless of where it's launched from, and so it plays nicely with
# PyInstaller-bundled resources.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST_JSON_PATH = os.path.join(BASE_DIR, "test_json", "test_output.json")

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
COLOR_KEYPOINT_CENTER = (255, 193, 7, 190)      # Amber/Yellow for center -- slightly translucent
COLOR_KEYPOINT_CORNER = (220, 53, 69, 190)      # Red for corner points -- slightly translucent
COLOR_COBB_LINE = (255, 87, 34, 200)            # Deep Orange for Cobb angle lines
COLOR_COBB_TEXT_BG = (255, 255, 255, 220)       # White for text background
COLOR_CSVL_LINE = (233, 30, 99, 210)            # Magenta/pink for the CSVL reference line

# Drag Handle Radius (screen pixels -- LandmarkHandleItem sets
# ItemIgnoresTransformations so these stay a constant on-screen size
# regardless of canvas zoom, rather than growing huge when zoomed in)
HANDLE_RADIUS = 4.0
ACTIVE_HANDLE_RADIUS = 5.5
