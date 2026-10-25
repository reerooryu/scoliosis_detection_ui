# App-wide constants.

# Application Window Configuration
APP_NAME = "Scoliosis Detection & Measurement UI"
# Version of this app. A release on GitHub is tagged "v" + APP_VERSION; the
# app tells the user when GitHub has a newer one (modules/updates.py).
APP_VERSION = "0.1.0"
GITHUB_REPO = "reerooryu/scoliosis_detection_ui"
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900

# Image types the app accepts. DICOM files are also recognised without an
# extension (modules/image_io.py).
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DICOM_EXTENSIONS = (".dcm", ".dicom")

# Where the AI model runs (changed in Settings):
#   "local"  -- on this computer, with a model file the user chooses;
#   "server" -- on an inference server (server.py) at an address.
DEFAULT_INFERENCE_MODE = "local"
INFERENCE_API_URL = "http://127.0.0.1:4000/predict"
INFERENCE_TIMEOUT = 120  # seconds, server mode only
# Name of the model file sent to testers. If it sits next to the app it is
# found without asking.
LOCAL_MODEL_FILENAME = "ScoliosisAI_model.pth"

# To move every user to one mode or server, set these in a release. They
# override what users saved in Settings; None means "use Settings".
# Example: FORCED_INFERENCE_MODE = "server"
#          FORCED_API_URL = "https://example.org/predict"
FORCED_INFERENCE_MODE = None
FORCED_API_URL = None

# Keypoint order within each vertebra.
KP_CENTER = 0
KP_TOP_LEFT = 1
KP_TOP_RIGHT = 2
KP_BOTTOM_LEFT = 3
KP_BOTTOM_RIGHT = 4

# Overlay colors as (R, G, B, A), each 0-255.
COLOR_VERTEBRA_OUTLINE = (0, 120, 215, 180)     # Blue, semi-transparent
COLOR_CORRIDOR_FILL = (0, 204, 150, 60)         # Emerald Green, low opacity (alpha = 60/255)
COLOR_KEYPOINT_CENTER = (255, 193, 7, 190)      # Amber/Yellow for center -- slightly translucent
COLOR_KEYPOINT_CORNER = (220, 53, 69, 190)      # Red for corner points -- slightly translucent
COLOR_CSVL_LINE = (233, 30, 99, 210)            # Magenta/pink for the CSVL reference line

# Color of each curve's Cobb lines and label, top curve first. Editable in
# Settings. A fifth curve reuses the first color.
CURVE_COLORS = ["#ff5722", "#ffd600", "#00e5ff", "#b388ff"]   # orange, yellow, cyan, violet

# Landmark handle radius in screen pixels (same size at any zoom level).
HANDLE_RADIUS = 4.0
ACTIVE_HANDLE_RADIUS = 5.5
