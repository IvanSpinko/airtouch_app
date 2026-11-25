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
# The first frames with a hand print one-off native warnings; silence until this many.
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
# The thumb counts as straight (L pose) when its tip is this far from the middle
# finger knuckle; bent in across the palm it is much closer.
THUMB_EXTENDED_RATIO = 0.7
# Click poses (thumb / index bent) must be seen this many frames in a row. Keep it
# small: a quick bend lasts only a few frames.
CLICK_POSE_FRAMES = 2
# After the hand (re)appears, click poses are ignored for this many frames.
ACQUIRE_GRACE_FRAMES = 6
# A finger is extended if wrist->tip is longer than wrist->pip by this factor.
FINGER_EXTENDED_RATIO = 1.15
# Fist: every fingertip must be within this distance of the palm center.
FIST_TIP_RATIO = 0.75
# Static poses (L, fist, palm, shaka...) must be stable for this many frames.
POSE_STABLE_FRAMES = 3
# Scroll pose ("two fingers") must be held longer, so the hand passing through it on
# the way to a click does not scroll the page.
SCROLL_STABLE_FRAMES = 6
# Scroll pose: index and middle fingertips held together (gap below this) ...
SCROLL_FINGERS_TOGETHER_RATIO = 0.35
# ... and the thumb away from both fingertips.
SCROLL_THUMB_AWAY_RATIO = 0.7
# The detector keeps its state through this many frames without a hand, so a single
# dropped frame does not end a drag.
HAND_LOST_GRACE_FRAMES = 5

# --- Gesture classifier (stage 3) -------------------------------------------
# Order matters: it is the label index used by the trained model.
GESTURE_CLASSES = (
    "none", "l_pose", "thumb_bent", "index_bent", "two_fingers", "fist", "open_palm", "shaka",
)
GESTURE_DATA_PATH = DATA_DIR / "gestures_v2.csv"
# Recordings of the first gesture set (pinches); unchanged poses are reused, see ml/migrate.py.
GESTURE_DATA_V1_PATH = DATA_DIR / "gestures.csv"
MIGRATE_V1_CLASSES = ("two_fingers", "fist", "open_palm")
GESTURE_MODEL_PT = MODELS_DIR / "gesture.pt"      # PyTorch checkpoint (training)
GESTURE_MODEL_NPZ = MODELS_DIR / "gesture.npz"    # numpy weights (runtime, no torch needed)
GESTURE_TEST_SPLIT = MODELS_DIR / "test_split.npz"
# Use the trained classifier at runtime when the model file exists.
USE_CLASSIFIER = True
# The model's pose is trusted only above this (smoothed) probability.
CLASSIFIER_MIN_PROB = 0.6
# Model probabilities are averaged over frames: new = a * frame + (1 - a) * old.
# 0.5 ~ the last 3 frames; lower = steadier but slower to react.
CLASSIFIER_SMOOTHING = 0.5

# Recording (ml/record.py)
RECORD_TARGET_PER_CLASS = 500
RECORD_KEYS = {ord(str(i + 1)): name for i, name in enumerate(GESTURE_CLASSES)}
RECORD_TOGGLE_KEY = ord(" ")
RECORD_UNDO_KEY = ord("u")
RECORD_UNDO_COUNT = 30

# Training (ml/train.py)
ML_SEED = 42
ML_VAL_FRACTION = 0.15
ML_TEST_FRACTION = 0.15
ML_HIDDEN = (128, 64)
ML_DROPOUT = 0.2
ML_EPOCHS = 80
ML_BATCH = 64
ML_LR = 1e-3
ML_WEIGHT_DECAY = 1e-4
ML_PATIENCE = 15
# Augmentation of normalized landmarks: random in-plane rotation (deg), scale, noise.
ML_AUG_ROTATION_DEG = 15.0
ML_AUG_SCALE = 0.1
ML_AUG_NOISE = 0.02
ML_TARGET_ACCURACY = 0.95

# --- Mouse actions ------------------------------------------------------------
# The cursor moves only in this pose ("clutch"): index + thumb straight, the rest
# folded. Any other pose leaves it where it is, like letting go of a mouse.
CURSOR_GESTURE = "l_pose"
# From the L pose: bending the thumb is the left button, bending the index finger
# the right button. The cursor follows the index knuckle, which neither moves.
LEFT_BUTTON_GESTURE = "thumb_bent"
RIGHT_BUTTON_GESTURE = "index_bent"
# A click counts when the hand returns to the L pose. After a fist (on its way in
# or out the hand passes through both bent poses) clicks are ignored this long.
AFTER_FIST_CLICK_GUARD_S = 0.5
# The click lands where the cursor was this long before the bend was recognized.
CLICK_LOOKBACK_S = 0.15
# How much cursor history to keep for the lookback, s.
CURSOR_HISTORY_S = 1.0
# Left button: a quick thumb bend clicks when the thumb straightens again - the
# button is never held down by it, so a false bend cannot drag or select. Keeping
# the thumb bent for DRAG_ARM_S presses the button ("DRAG READY"); moving the hand
# DRAG_START_PX (smoothed cursor) then drags, straightening the thumb drops.
DRAG_START_PX = 60
DRAG_ARM_S = 0.4
# While a click pose is building up (strength at least this), the cursor stops, so
# the click lands where you aimed.
CURSOR_FREEZE_CLICK_LEVEL = 0.5
# Safety: a left button held this long without dragging is released (prevents a
# stuck button from an accidental press).
LEFT_HOLD_STILL_MAX_S = 2.0
# A bend must last at least this long to click (filters passing flickers).
CLICK_MIN_HOLD_S = 0.06
# After a click the cursor stays frozen this long (unless the hand moves more than
# DRAG_START_PX), so two quick clicks land in one spot and make a double click.
CLICK_SETTLE_S = 0.35
# Scroll mode:
# "auto": like pressing the mouse wheel — the spot where the two-finger pose starts
#         is the anchor; moving the hand above / below it scrolls continuously, the
#         further away the faster, until the fingers are opened. Default.
# "drag": the page follows the hand movement (move up = scroll up, stop = stop).
SCROLL_MODE = "auto"
# Auto scroll: no scrolling within this distance of the anchor (fraction of frame height).
SCROLL_AUTO_DEADZONE = 0.03
# Auto scroll speed: wheel ticks per second per frame-height of offset beyond the dead zone.
SCROLL_AUTO_SPEED = 80.0
# Auto scroll speed limit, wheel ticks per second.
SCROLL_AUTO_MAX_TICKS_S = 25.0
# Drag scroll: wheel ticks per full frame height of hand movement.
SCROLL_TICKS_PER_FRAME = 25.0
# Fractional ticks are sent for smooth scrolling once at least this many accumulate.
SCROLL_MIN_TICKS = 0.1
# Drag scroll: the hand must move this much (fraction of frame height) before scrolling starts,
# so entering the pose does not nudge the page.
SCROLL_START_RATIO = 0.02
# True = reverses the scroll direction (hand up scrolls down) in both modes.
SCROLL_INVERT = False
# Holding a fist this long toggles pause.
FIST_HOLD_S = 0.8
# Without a hand for this long, any held button is released.
HAND_LOST_RELEASE_S = 0.5

# Point the cursor follows: weighted mean of landmarks {index: weight}.
# The base knuckle of the index finger moves with the hand but not with finger
# bending, so pinching and finger jitter do not shift the cursor. For the old
# fingertip behaviour use {LM_INDEX_TIP: 1.0}; a blend like
# {LM_INDEX_MCP: 0.7, LM_INDEX_TIP: 0.3} trades some stability for finer control.
CURSOR_POINT_WEIGHTS = {LM_INDEX_MCP: 1.0}

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
ACCEL_MIN_GAIN = 0.3
ACCEL_MAX_GAIN = 3.0
# Hand speed (in absolute-mode screen px/s) where acceleration starts / saturates.
ACCEL_LOW_SPEED_PX_S = 250.0
ACCEL_HIGH_SPEED_PX_S = 2000.0

# --- Cursor smoothing (One Euro Filter, in screen pixels) ----------------------
# min_cutoff — lower = stronger jitter smoothing at rest.
# beta — higher = less lag during fast movement.
FILTER_MIN_CUTOFF = 0.3
FILTER_BETA = 0.02
FILTER_D_CUTOFF = 1.0
# Cursor moves smaller than this (px) are ignored — removes micro-jitter.
CURSOR_DEADZONE_PX = 3.0

# Outlier rejection: MediaPipe sometimes makes the landmarks jump for a frame or
# two (e.g. when re-detecting the hand). A hand "moving" faster than this
# (frame heights per second) is ignored; if the jump persists for more than
# OUTLIER_CONFIRM_FRAMES frames it is real, and the cursor re-anchors there
# without jumping.
OUTLIER_SPEED = 6.0
OUTLIER_CONFIRM_FRAMES = 2

# --- Cursor animation ---------------------------------------------------------
# The camera gives ~30 targets per second; a separate thread glides the real cursor
# towards the latest target at a higher rate, so motion looks continuous instead of
# jumping several pixels every frame.
CURSOR_ANIMATION_HZ = 144
# Time constant of the glide (exponential approach), s. Higher = smoother, more lag.
CURSOR_GLIDE_TAU_S = 0.06
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

# --- On-screen keyboard (stage 4) ---------------------------------------------
OSK_PATH = r"C:\Windows\System32\osk.exe"
OSK_WINDOW_CLASS = "OSKMainClass"
OSK_REGISTRY_KEY = r"Software\Microsoft\Osk"
# Enlarged keyboard: share of the screen, centered at the bottom.
OSK_WIDTH_FRACTION = 0.95
OSK_HEIGHT_FRACTION = 0.48
OSK_BOTTOM_MARGIN_FRACTION = 0.02
OSK_WORD_PREDICTION = True
OSK_MODE_CLICK = 1   # osk.exe "Mode" value: click on keys (not hover / scan)
OSK_WAIT_S = 5.0     # how long to wait for the window to appear / disappear
OSK_POLL_S = 0.1
# --check: where the close (X) button is, from the window's top-right corner, px
OSK_CLOSE_BUTTON_DX = 25
OSK_CLOSE_BUTTON_DY = 18
OSK_CHECK_SETTLE_S = 0.5
OSK_CHECK_RESULT_S = 1.5
# Holding this pose (only thumb + pinky out) opens / closes the keyboard.
KEYBOARD_GESTURE = "shaka"
KEYBOARD_TOGGLE_HOLD_S = 1.0
# osk.exe only accepts our clicks when AirTouch runs as administrator; ask for it
# (UAC prompt) on start. --no-admin skips it (keyboard clicks will not work).
RUN_AS_ADMIN = True

# Typing test (python -m gesture_control.typing_test)
CHARS_PER_WORD = 5      # standard WPM definition
TARGET_WPM = 10         # stage 4 target from the spec
TYPING_TEST_PHRASES = (
    "hello world",
    "the quick brown fox",
    "gesture control works",
)

# --- Windows API constants (keyboard) -------------------------------------------
BASE_DPI = 96
SW_SHOWNORMAL = 1
SHELL_EXECUTE_OK = 32   # ShellExecute returns > 32 on success
WM_CLOSE = 0x0010

# --- Hotkeys ------------------------------------------------------------------
# pynput.keyboard.GlobalHotKeys format.
HOTKEY_TOGGLE = "<ctrl>+<alt>+g"

# --- Session log (python -m gesture_control.main --log) ------------------------
SESSION_LOG_NAME = "session-%Y%m%d-%H%M%S.csv"
SESSION_MARK_KEY = ord("m")
# ml/replay.py: how many seconds before each MARK to show.
REPLAY_MARK_WINDOW_S = 3.0

# --- Preview window (debug, until the PyQt overlay exists) --------------------
SHOW_PREVIEW = True
# Show raw rule features in the preview (for tuning THUMB_EXTENDED_RATIO etc.).
PREVIEW_SHOW_FEATURES = False
# Pinch strength bars (bottom-left of the preview): size in px and colors.
PREVIEW_METER_WIDTH = 160
PREVIEW_METER_HEIGHT = 14
PREVIEW_METER_GAP = 8
PREVIEW_METER_MARGIN = 12
COLOR_METER_BG = (60, 60, 60)
COLOR_METER_FILL = (200, 200, 200)
COLOR_METER_ACTIVE = (0, 255, 0)
COLOR_METER_ON = (0, 255, 255)
COLOR_METER_OFF = (0, 140, 255)
# How long the last mouse action ("LEFT CLICK", ...) stays on screen, s.
PREVIEW_EVENT_FLASH_S = 0.7
PREVIEW_EVENT_FONT_SCALE = 1.2
PREVIEW_EVENT_THICKNESS = 3
COLOR_EVENT = (0, 255, 255)
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
COLOR_SCROLL_ANCHOR = (0, 165, 255)
COLOR_SCROLL_DEADZONE = (0, 100, 160)
# Preview warns when the hand is closer than this (fraction of the frame) to an edge.
EDGE_WARN_MARGIN = 0.05
EDGE_WARN_THICKNESS = 12
COLOR_EDGE_WARN = (0, 0, 255)

# FPS counter smoothing (exponential moving average).
FPS_SMOOTHING = 0.9
