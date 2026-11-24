"""Hand tracking: MediaPipe Hand Landmarker -> 21 landmarks.

The ``hand_landmarker.task`` model is downloaded once into ``models/`` on first run.
Frames themselves never leave the machine — processing is fully local.

Check: ``python -m gesture_control.tracker`` shows a window with hand landmarks.
"""

from __future__ import annotations

import contextlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from gesture_control import config


@contextlib.contextmanager
def native_stderr_silenced():
    """Temporarily points file descriptor 2 to the null device.

    MediaPipe / LiteRT print INFO lines from C++ straight to fd 2 when the model is
    created, on the first frames and on close; Python-level logging settings and
    env vars do not reach them.
    """
    if not config.SILENCE_NATIVE_LOGS:
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)

# Landmark connections used to draw the hand skeleton.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
)


@dataclass(frozen=True)
class Hand:
    """Tracking result for one hand.

    landmarks — array (21, 3): x, y normalized to frame width/height (0..1),
    z — relative depth (smaller = closer to the camera).
    handedness — "Left" or "Right" (for a mirrored frame it matches the real hand).
    """

    landmarks: np.ndarray
    handedness: str
    score: float

    def point(self, index: int) -> tuple[float, float]:
        return float(self.landmarks[index, 0]), float(self.landmarks[index, 1])

    def cursor_point(
        self, weights: dict[int, float] = config.CURSOR_POINT_WEIGHTS
    ) -> tuple[float, float]:
        """Weighted mean of landmarks the cursor follows (see CURSOR_POINT_WEIGHTS)."""
        total = sum(weights.values())
        x = sum(self.landmarks[i, 0] * w for i, w in weights.items()) / total
        y = sum(self.landmarks[i, 1] * w for i, w in weights.items()) / total
        return float(x), float(y)


def ensure_model(
    path: Path = config.HAND_MODEL_PATH, url: str = config.HAND_MODEL_URL
) -> Path:
    """Downloads the MediaPipe model if it is not present locally."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {path} ...")
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=config.HAND_MODEL_DOWNLOAD_TIMEOUT_S) as resp:
        tmp.write_bytes(resp.read())
    tmp.replace(path)
    return path


class HandTracker:
    """Wrapper around HandLandmarker in VIDEO mode."""

    def __init__(self, model_path: Path | None = None) -> None:
        model_path = ensure_model() if model_path is None else model_path
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        with native_stderr_silenced():
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        self._frames = 0

    def close(self) -> None:
        with native_stderr_silenced():
            self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def process(self, frame_bgr: np.ndarray, timestamp_s: float) -> Hand | None:
        """Finds a hand in the frame. Returns None if there is no hand."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps.
        ts_ms = max(int(timestamp_s * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        # The one-off native warnings appear when the landmark model first runs,
        # i.e. on the first frames *with a hand*, which may be long after start-up.
        if self._frames < config.SILENCED_WARMUP_FRAMES:
            with native_stderr_silenced():
                result = self._landmarker.detect_for_video(image, ts_ms)
            if result.hand_landmarks:
                self._frames += 1
        else:
            result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.hand_landmarks:
            return None
        lms = np.array(
            [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]], dtype=np.float32
        )
        category = result.handedness[0][0]
        return Hand(landmarks=lms, handedness=category.category_name, score=category.score)


def near_edges(hand: Hand, margin: float = config.EDGE_WARN_MARGIN) -> set[str]:
    """Frame edges ("left", "right", "top", "bottom") the hand is about to leave.

    Near an edge part of the hand is cut off and MediaPipe starts losing it.
    """
    xs, ys = hand.landmarks[:, 0], hand.landmarks[:, 1]
    sides = set()
    if xs.min() < margin:
        sides.add("left")
    if xs.max() > 1.0 - margin:
        sides.add("right")
    if ys.min() < margin:
        sides.add("top")
    if ys.max() > 1.0 - margin:
        sides.add("bottom")
    return sides


def draw_hand(frame: np.ndarray, hand: Hand) -> None:
    """Draws the hand skeleton over the frame (in place)."""
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in hand.landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], config.COLOR_BONE, config.PREVIEW_LINE_THICKNESS)
    for p in pts:
        cv2.circle(frame, p, config.PREVIEW_LANDMARK_RADIUS, config.COLOR_LANDMARK, cv2.FILLED)


def _demo() -> None:
    from gesture_control.camera import Camera, FpsCounter

    fps = FpsCounter()
    with Camera() as cam, HandTracker() as tracker:
        while cam.is_running:
            item = cam.read(timeout=1.0)
            if item is None:
                continue
            frame, ts = item
            hand = tracker.process(frame, ts)
            frame = frame.copy()
            label = "no hand"
            if hand is not None:
                draw_hand(frame, hand)
                label = f"{hand.handedness} {hand.score:.2f}"
            cv2.putText(
                frame, f"FPS: {fps.tick():.1f}  {label}", config.PREVIEW_TEXT_ORIGIN,
                cv2.FONT_HERSHEY_SIMPLEX, config.PREVIEW_FONT_SCALE,
                config.COLOR_TEXT_ACTIVE, config.PREVIEW_FONT_THICKNESS,
            )
            cv2.imshow("tracker", frame)
            if cv2.waitKey(config.PREVIEW_WAIT_KEY_MS) & 0xFF in config.PREVIEW_QUIT_KEYS:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
