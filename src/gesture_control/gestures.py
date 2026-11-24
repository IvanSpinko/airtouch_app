"""Gesture recognition from the 21 hand landmarks.

Stage 2: rule-based detector (pinches by distance thresholds with hysteresis, static
poses by which fingers are extended). Stage 3 adds a trained classifier on top.

All measurements are normalized by palm size, so they do not depend on the distance
to the camera, and they are symmetric, so left and right hands work the same way.

Check: ``python -m gesture_control.gestures`` shows the recognized gesture live.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from gesture_control import config


class Gesture(str, Enum):
    NONE = "none"
    POINT = "point"                # index finger extended — move the cursor
    PINCH_INDEX = "pinch_index"    # thumb + index — left button
    PINCH_MIDDLE = "pinch_middle"  # thumb + middle — right click
    TWO_FINGERS = "two_fingers"    # index + middle extended — scroll
    FIST = "fist"                  # pause
    OPEN_PALM = "open_palm"        # reserved for the on-screen keyboard (stage 4)


@dataclass(frozen=True)
class HandFeatures:
    """Scale-invariant measurements of one hand pose."""

    pinch_index: float
    pinch_middle: float
    extended: dict[str, bool]

    @property
    def extended_count(self) -> int:
        return sum(self.extended.values())


def _xy(landmarks: np.ndarray, aspect: float) -> np.ndarray:
    pts = np.asarray(landmarks, dtype=np.float64)[:, :2].copy()
    pts[:, 0] *= aspect
    return pts


def compute_features(landmarks: np.ndarray, aspect: float = config.FRAME_ASPECT) -> HandFeatures:
    pts = _xy(landmarks, aspect)

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(pts[a] - pts[b]))

    palm = max(dist(config.LM_WRIST, config.LM_MIDDLE_MCP), 1e-6)
    extended = {
        name: dist(config.LM_WRIST, tip) > dist(config.LM_WRIST, pip) * config.FINGER_EXTENDED_RATIO
        for name, (pip, tip) in config.FINGER_JOINTS.items()
    }
    return HandFeatures(
        pinch_index=dist(config.LM_THUMB_TIP, config.LM_INDEX_TIP) / palm,
        pinch_middle=dist(config.LM_THUMB_TIP, config.LM_MIDDLE_TIP) / palm,
        extended=extended,
    )


def classify_pose(f: HandFeatures) -> Gesture:
    """Static pose from finger states (no pinch logic, no history)."""
    e = f.extended
    if f.extended_count == 0:
        return Gesture.FIST
    if f.extended_count == 4 and f.pinch_index > config.OPEN_PALM_THUMB_RATIO:
        return Gesture.OPEN_PALM
    if e["index"] and e["middle"] and not e["ring"] and not e["pinky"]:
        return Gesture.TWO_FINGERS
    if e["index"]:
        return Gesture.POINT
    return Gesture.NONE


class GestureDetector:
    """Stateful detector: pinch hysteresis + debouncing of static poses."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._pinch: Gesture | None = None
        self._pinch_candidate: Gesture | None = None
        self._pinch_frames = 0
        self._pose = Gesture.NONE
        self._pose_candidate = Gesture.NONE
        self._pose_frames = 0

    def _update_pinch(self, f: HandFeatures) -> Gesture | None:
        if self._pinch is Gesture.PINCH_INDEX:
            if f.pinch_index < config.PINCH_OFF_RATIO:
                return self._pinch
            self._pinch = None
        elif self._pinch is Gesture.PINCH_MIDDLE:
            if f.pinch_middle < config.PINCH_OFF_RATIO:
                return self._pinch
            self._pinch = None

        candidate = None
        if min(f.pinch_index, f.pinch_middle) < config.PINCH_ON_RATIO:
            candidate = (
                Gesture.PINCH_INDEX if f.pinch_index <= f.pinch_middle else Gesture.PINCH_MIDDLE
            )
        if candidate is not None and candidate is self._pinch_candidate:
            self._pinch_frames += 1
        else:
            self._pinch_candidate = candidate
            self._pinch_frames = 1 if candidate is not None else 0
        if candidate is not None and self._pinch_frames >= config.PINCH_CONFIRM_FRAMES:
            self._pinch = candidate
            self._pinch_candidate = None
            self._pinch_frames = 0
        return self._pinch

    def _update_pose(self, f: HandFeatures) -> Gesture:
        raw = classify_pose(f)
        if raw is self._pose_candidate:
            self._pose_frames += 1
        else:
            self._pose_candidate = raw
            self._pose_frames = 1
        if self._pose_frames >= config.POSE_STABLE_FRAMES:
            self._pose = raw
        return self._pose

    def update(self, landmarks: np.ndarray | None) -> Gesture:
        if landmarks is None:
            self.reset()
            return Gesture.NONE
        f = compute_features(landmarks)
        pose = self._update_pose(f)
        # A fist brings all fingertips close to the thumb, so it must never count as
        # a pinch — checked on the raw current frame, not the debounced pose, so a
        # click cannot slip through while the fist is still being confirmed.
        if f.extended_count == 0:
            self._pinch = self._pinch_candidate = None
            self._pinch_frames = 0
            return pose
        pinch = self._update_pinch(f)
        return pinch if pinch is not None else pose


def _demo() -> None:
    import cv2

    from gesture_control.camera import Camera
    from gesture_control.tracker import HandTracker, draw_hand

    detector = GestureDetector()
    with Camera() as cam, HandTracker() as tracker:
        while cam.is_running:
            item = cam.read(timeout=1.0)
            if item is None:
                continue
            frame, ts = item
            hand = tracker.process(frame, ts)
            gesture = detector.update(hand.landmarks if hand else None)
            frame = frame.copy()
            text = gesture.value
            if hand is not None:
                draw_hand(frame, hand)
                f = compute_features(hand.landmarks)
                text += f"  pi={f.pinch_index:.2f} pm={f.pinch_middle:.2f}"
            cv2.putText(
                frame, text, config.PREVIEW_TEXT_ORIGIN, cv2.FONT_HERSHEY_SIMPLEX,
                config.PREVIEW_FONT_SCALE, config.COLOR_TEXT_ACTIVE, config.PREVIEW_FONT_THICKNESS,
            )
            cv2.imshow("gestures", frame)
            if cv2.waitKey(config.PREVIEW_WAIT_KEY_MS) & 0xFF in config.PREVIEW_QUIT_KEYS:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
