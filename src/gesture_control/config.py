"""All application thresholds and parameters.

Project rule: no magic numbers in other modules — they live here only.
"""

from pathlib import Path

# --- Paths --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# --- Camera -------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
# Mirror the frame so moving the hand right moves the cursor right.
CAMERA_MIRROR = True
# Consecutive failed reads before the camera is considered lost.
CAMERA_MAX_FAILED_READS = 30
# How long to wait for the first frame, s.
CAMERA_FIRST_FRAME_TIMEOUT_S = 5.0

# --- Hand tracking (MediaPipe Hand Landmarker) --------------------------------
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_DOWNLOAD_TIMEOUT_S = 60
MAX_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.6
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
NUM_LANDMARKS = 21
# Environment variables that quiet native MediaPipe / TFLite logging.
NATIVE_LOG_LEVELS = {"GLOG_minloglevel": "2", "TF_CPP_MIN_LOG_LEVEL": "2"}

# MediaPipe hand landmark indices.
LM_WRIST = 0
LM_THUMB_TIP = 4
LM_INDEX_MCP = 5
LM_INDEX_TIP = 8
LM_MIDDLE_TIP = 12
LM_RING_TIP = 16
LM_PINKY_TIP = 20

# Landmark the cursor follows.
CURSOR_LANDMARK = LM_INDEX_TIP

# --- Work zone ----------------------------------------------------------------
# Rectangle in normalized frame coordinates (0..1) that is stretched over the whole
# screen. A small zone = less arm fatigue and the screen corners are easy to reach.
WORK_ZONE_LEFT = 0.20
WORK_ZONE_RIGHT = 0.80
WORK_ZONE_TOP = 0.15
WORK_ZONE_BOTTOM = 0.65
# Sensitivity: >1 shrinks the work zone around its center (faster cursor).
CURSOR_SENSITIVITY = 1.0

# --- Cursor smoothing (One Euro Filter, in screen pixels) ----------------------
# min_cutoff — lower = stronger jitter smoothing at rest.
# beta — higher = less lag during fast movement.
FILTER_MIN_CUTOFF = 0.5
FILTER_BETA = 0.02
FILTER_D_CUTOFF = 1.0
# Cursor moves smaller than this (px) are ignored — removes micro-jitter.
CURSOR_DEADZONE_PX = 2.5

# --- Cursor animation ---------------------------------------------------------
# The camera gives ~30 targets per second; a separate thread glides the real cursor
# towards the latest target at a higher rate, so motion looks continuous instead of
# jumping several pixels every frame.
CURSOR_ANIMATION_HZ = 144
# Time constant of the glide (exponential approach), s. Higher = smoother, more lag.
CURSOR_GLIDE_TAU_S = 0.025
# Closer than this to the target (px) — snap and stop moving.
CURSOR_SNAP_PX = 0.5
# If the hand is lost for longer than this, the filter is reset so the cursor
# does not glide in from a stale position.
HAND_LOST_RESET_S = 0.3

# --- Windows API constants ----------------------------------------------------
DPI_AWARENESS_PER_MONITOR = 2  # PROCESS_PER_MONITOR_DPI_AWARE
SM_CXSCREEN = 0
SM_CYSCREEN = 1
# Pause between steps of the ``python -m gesture_control.mouse`` demo, s.
MOUSE_DEMO_STEP_S = 0.7

# --- Hotkeys ------------------------------------------------------------------
# pynput.keyboard.GlobalHotKeys format.
HOTKEY_TOGGLE = "<ctrl>+<alt>+g"

# --- Preview window (debug, until the PyQt overlay exists) --------------------
SHOW_PREVIEW = True
PREVIEW_WINDOW_NAME = "AirTouch preview"
PREVIEW_QUIT_KEYS = (ord("q"), 27)  # q or Esc
PREVIEW_WAIT_KEY_MS = 1
PREVIEW_FONT_SCALE = 0.6
PREVIEW_FONT_THICKNESS = 2
PREVIEW_TEXT_ORIGIN = (10, 25)
PREVIEW_TEXT_LINE_HEIGHT = 25
PREVIEW_LANDMARK_RADIUS = 4
PREVIEW_CURSOR_RADIUS = 8
PREVIEW_LINE_THICKNESS = 2
# Colors in BGR.
COLOR_LANDMARK = (0, 255, 0)
COLOR_BONE = (255, 255, 255)
COLOR_CURSOR_POINT = (0, 0, 255)
COLOR_ZONE = (255, 200, 0)
COLOR_TEXT_ACTIVE = (0, 255, 0)
COLOR_TEXT_PAUSED = (0, 0, 255)

# FPS counter smoothing (exponential moving average).
FPS_SMOOTHING = 0.9
