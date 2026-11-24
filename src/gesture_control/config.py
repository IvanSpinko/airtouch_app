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
# Lower values keep tracking the hand through blur and odd angles instead of
# dropping it (which freezes the cursor); a single user is assumed, so false
# detections of other hands are not a concern.
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.4
MIN_TRACKING_CONFIDENCE = 0.4
NUM_LANDMARKS = 21
# Hide MediaPipe / LiteRT native INFO logs on start-up and shutdown.
SILENCE_NATIVE_LOGS = True
# The first frames also print one-off native warnings; silence that many.
SILENCED_WARMUP_FRAMES = 3

# MediaPipe hand landmark indices.
LM_WRIST = 0
LM_THUMB_TIP = 4
LM_INDEX_MCP = 5
LM_INDEX_PIP = 6
LM_INDEX_TIP = 8
LM_MIDDLE_MCP = 9
LM_MIDDLE_PIP = 10
LM_MIDDLE_TIP = 12
LM_RING_MCP = 13
LM_RING_PIP = 14
LM_RING_TIP = 16
LM_PINKY_MCP = 17
LM_PINKY_PIP = 18
LM_PINKY_TIP = 20
# (pip, tip) per finger, thumb excluded.
FINGER_JOINTS = {
    "index": (LM_INDEX_PIP, LM_INDEX_TIP),
    "middle": (LM_MIDDLE_PIP, LM_MIDDLE_TIP),
    "ring": (LM_RING_PIP, LM_RING_TIP),
    "pinky": (LM_PINKY_PIP, LM_PINKY_TIP),
}
# Landmarks x/y are normalized separately by frame width and height; distances are
# measured after scaling x by this aspect ratio so they are not distorted.
FRAME_ASPECT = CAMERA_WIDTH / CAMERA_HEIGHT
# Weight of MediaPipe depth (z) in 3D distances. Depth lets a finger pointed at the
# camera still count as extended; 0 would fall back to pure 2D.
DEPTH_WEIGHT = 1.0
# Landmarks whose mean is the palm center: wrist and the four finger MCPs.
PALM_LANDMARKS = (LM_WRIST, LM_INDEX_MCP, LM_MIDDLE_MCP, LM_RING_MCP, LM_PINKY_MCP)

# --- Rule-based gestures ------------------------------------------------------
# All distances are relative to the palm size (wrist -> middle finger MCP),
# so thresholds do not depend on how far the hand is from the camera.
# Pinch hysteresis: starts below ON, ends above OFF.
PINCH_ON_RATIO = 0.25
PINCH_OFF_RATIO = 0.38
# Frames a pinch must be seen before it counts (filters single-frame glitches).
PINCH_CONFIRM_FRAMES = 2
# A finger is extended if wrist->tip is longer than wrist->pip by this factor.
FINGER_EXTENDED_RATIO = 1.15
# Fist: every fingertip must be within this distance of the palm center.
FIST_TIP_RATIO = 0.75
# Open palm also requires the thumb to be far from the index fingertip.
OPEN_PALM_THUMB_RATIO = 0.6
# Static poses (fist, two fingers, palm, point) must be stable for this many frames.
POSE_STABLE_FRAMES = 3
# Scroll pose ("two fingers") must be held longer, so the hand passing through it on
# the way to a click does not scroll the page.
SCROLL_STABLE_FRAMES = 6
# Scroll pose: index and middle fingertips held together (gap below this) ...
SCROLL_FINGERS_TOGETHER_RATIO = 0.35
# ... and the thumb away from the middle fingertip (not getting ready to pinch).
SCROLL_THUMB_AWAY_RATIO = 0.7
# The detector keeps its state through this many frames without a hand, so a single
# dropped frame does not end a drag.
HAND_LOST_GRACE_FRAMES = 5

# --- Mouse actions ------------------------------------------------------------
# Which pinch acts as which mouse button ("pinch_index" or "pinch_middle").
# Thumb + middle for the left button keeps the index finger — and so the cursor —
# still while clicking.
LEFT_BUTTON_GESTURE = "pinch_middle"
RIGHT_BUTTON_GESTURE = "pinch_index"
# Closing the pinch drags the fingertip; the click lands where the cursor was this
# long before the pinch was recognized.
CLICK_LOOKBACK_S = 0.15
# How much cursor history to keep for the lookback, s.
CURSOR_HISTORY_S = 1.0
# While pinched, the cursor stays put; it becomes a drag only once the pinch has
# been held at least DRAG_MIN_HOLD_S AND the smoothed cursor moved DRAG_START_PX.
# A quick pinch is therefore always a clean click, never a tiny drag/selection.
DRAG_START_PX = 60
DRAG_MIN_HOLD_S = 0.3
# After a click the cursor stays frozen this long (unless the hand moves more than
# DRAG_START_PX), so two quick pinches land in one spot and make a double click.
CLICK_SETTLE_S = 0.35
# Scroll: wheel ticks per full frame height of hand movement. Fractional ticks are
# sent for smooth scrolling once at least SCROLL_MIN_TICKS accumulate.
SCROLL_TICKS_PER_FRAME = 25.0
SCROLL_MIN_TICKS = 0.1
# The hand must move this much (fraction of frame height) before scrolling starts,
# so entering the pose does not nudge the page.
SCROLL_START_RATIO = 0.02
# True = moving the hand up scrolls down (like dragging the page).
SCROLL_INVERT = False
# Holding a fist this long toggles pause.
FIST_HOLD_S = 0.8
# Without a hand for this long, any held button is released.
HAND_LOST_RELEASE_S = 0.5

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

# --- Cursor mode and pointer acceleration -------------------------------------
# "relative": the cursor moves by hand movement with acceleration, like a mouse
#             (precise when slow, fast when flicked). Default.
# "absolute": the work zone is stretched over the screen 1:1.
CURSOR_MODE = "relative"
# Gain applied to hand movement (1.0 = the same speed as absolute mode).
# Slow hand movement gets ACCEL_MIN_GAIN, fast movement ACCEL_MAX_GAIN.
ACCEL_MIN_GAIN = 0.25
ACCEL_MAX_GAIN = 1.6
# Hand speed (in absolute-mode screen px/s) where acceleration starts / saturates.
ACCEL_LOW_SPEED_PX_S = 300.0
ACCEL_HIGH_SPEED_PX_S = 3000.0

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
