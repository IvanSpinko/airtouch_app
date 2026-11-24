"""Gesture recognition from the 21 hand landmarks.

Stage 2: rule-based detector (pinches by distance thresholds with hysteresis, static
poses by which fingers are extended). Stage 3 adds a trained classifier on top.

All measurements are 3D, normalized by palm size, so they do not depend on the distance
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
    PINCH_INDEX = "pinch_index"    # thumb + index (right button by default)
    PINCH_MIDDLE = "pinch_middle"  # thumb + middle (left button by default)
    TWO_FINGERS = "two_fingers"    # index + middle extended together — scroll
    FIST = "fist"                  # pause
    OPEN_PALM = "open_palm"        # reserved for the on-screen keyboard (stage 4)


@dataclass(frozen=True)
class HandFeatures:
    """Scale-invariant measurements of one hand pose."""

    pinch_index: float
    pinch_middle: float
    extended: dict[str, bool]
    # Largest fingertip distance from the palm center (relative to palm size).
    max_tip_spread: float
    # Distance between index and middle fingertips.
    index_middle_gap: float = 0.0

    @property
    def extended_count(self) -> int:
        return sum(self.extended.values())

    @property
    def is_fist(self) -> bool:
        """No finger extended AND every fingertip is curled in close to the palm.

        The second check stops a finger pointed at the camera (short in 2D) from
        being taken for a fist and freezing the cursor.
        """
        return self.extended_count == 0 and self.max_tip_spread < config.FIST_TIP_RATIO


def _xyz(landmarks: np.ndarray, aspect: float) -> np.ndarray:
    """Landmarks in isotropic units: x is rescaled by the frame aspect ratio, and z
    (MediaPipe depth, same scale as x) too, so 3D distances are not distorted."""
    pts = np.asarray(landmarks, dtype=np.float64)[:, :3].copy()
    pts[:, 0] *= aspect
    pts[:, 2] *= aspect * config.DEPTH_WEIGHT
    return pts


def compute_features(landmarks: np.ndarray, aspect: float = config.FRAME_ASPECT) -> HandFeatures:
    pts = _xyz(landmarks, aspect)

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(pts[a] - pts[b]))

    palm = max(dist(config.LM_WRIST, config.LM_MIDDLE_MCP), 1e-6)
    extended = {
        name: dist(config.LM_WRIST, tip) > dist(config.LM_WRIST, pip) * config.FINGER_EXTENDED_RATIO
        for name, (pip, tip) in config.FINGER_JOINTS.items()
    }
    center = pts[list(config.PALM_LANDMARKS)].mean(axis=0)
    spread = max(
        float(np.linalg.norm(pts[tip] - center)) for _, tip in config.FINGER_JOINTS.values()
    ) / palm
    return HandFeatures(
        pinch_index=dist(config.LM_THUMB_TIP, config.LM_INDEX_TIP) / palm,
        pinch_middle=dist(config.LM_THUMB_TIP, config.LM_MIDDLE_TIP) / palm,
        extended=extended,
        max_tip_spread=spread,
        index_middle_gap=dist(config.LM_INDEX_TIP, config.LM_MIDDLE_TIP) / palm,
    )


def normalize_landmarks(
    landmarks: np.ndarray, handedness: str = "Right", aspect: float = config.FRAME_ASPECT
) -> np.ndarray:
    """Pose-only feature vector (63 floats) for the classifier.

    Removes everything that is not the hand shape: position (wrist at the origin),
    size (palm length = 1), in-plane rotation (wrist -> middle MCP points up) and
    handedness (left hands are mirrored to look like right hands).
    """
    pts = np.asarray(landmarks, dtype=np.float64)[:, :3].copy()
    pts[:, 0] *= aspect
    pts[:, 2] *= aspect
    pts -= pts[config.LM_WRIST]
    if handedness == "Left":
        pts[:, 0] = -pts[:, 0]
    vx, vy = pts[config.LM_MIDDLE_MCP, 0], pts[config.LM_MIDDLE_MCP, 1]
    palm = max(float(np.hypot(vx, vy)), 1e-6)
    # rotate so that (vx, vy) maps to (0, -palm): fingers point "up" in image coords
    angle = np.arctan2(vx, -vy)
    cos, sin = np.cos(angle), np.sin(angle)
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    pts[:, 0] = cos * x + sin * y
    pts[:, 1] = -sin * x + cos * y
    pts /= palm
    return pts.reshape(-1).astype(np.float32)


class GestureClassifier:
    """Runtime MLP inference in numpy (weights exported by ml/train.py).

    numpy keeps the app light: no PyTorch needed to run it or in the .exe.
    """

    def __init__(self, path=config.GESTURE_MODEL_NPZ) -> None:
        data = np.load(path, allow_pickle=False)
        self.classes = [Gesture(c) for c in data["classes"]]
        n = int(data["num_layers"])
        self.layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n)]

    @classmethod
    def load_if_available(cls, path=config.GESTURE_MODEL_NPZ) -> "GestureClassifier | None":
        return cls(path) if config.USE_CLASSIFIER and path.exists() else None

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        h = np.asarray(features, dtype=np.float32)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        h = h - h.max(axis=-1, keepdims=True)
        e = np.exp(h)
        return e / e.sum(axis=-1, keepdims=True)

    def predict(self, landmarks: np.ndarray, handedness: str = "Right") -> dict[Gesture, float]:
        probs = self.predict_proba(normalize_landmarks(landmarks, handedness))
        return dict(zip(self.classes, map(float, probs)))


def classify_pose(f: HandFeatures) -> Gesture:
    """Static pose from finger states (no pinch logic, no history)."""
    e = f.extended
    if f.is_fist:
        return Gesture.FIST
    if f.extended_count == 4 and f.pinch_index > config.OPEN_PALM_THUMB_RATIO:
        return Gesture.OPEN_PALM
    if (
        e["index"] and e["middle"] and not e["ring"] and not e["pinky"]
        # Fingers together and thumb away: otherwise this is the hand getting ready
        # for a thumb + middle pinch, which looks the same by finger states alone.
        and f.index_middle_gap < config.SCROLL_FINGERS_TOGETHER_RATIO
        and f.pinch_middle > config.SCROLL_THUMB_AWAY_RATIO
    ):
        return Gesture.TWO_FINGERS
    if e["index"]:
        return Gesture.POINT
    return Gesture.NONE


class GestureDetector:
    """Stateful detector: pinch hysteresis + debouncing of static poses."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._lost_frames = 0
        self._seen_frames = 0
        self._release_frames = 0
        self._last = Gesture.NONE
        self._pinch: Gesture | None = None
        self._pinch_candidate: Gesture | None = None
        self._pinch_frames = 0
        self._pose = Gesture.NONE
        self._pose_candidate = Gesture.NONE
        self._pose_frames = 0

    def _update_pinch(self, f: HandFeatures) -> Gesture | None:
        if self._pinch is not None:
            dist = f.pinch_index if self._pinch is Gesture.PINCH_INDEX else f.pinch_middle
            if dist < config.PINCH_OFF_RATIO:
                self._release_frames = 0
                return self._pinch
            # Release only after a few open frames, so a flicker mid-drag does not
            # drop the item or turn one click into two.
            self._release_frames += 1
            if self._release_frames < config.PINCH_RELEASE_FRAMES:
                return self._pinch
            self._pinch = None
            self._release_frames = 0

        candidate = None
        # Right after the hand appears its landmarks are unreliable: no new clicks.
        acquiring = self._seen_frames <= config.ACQUIRE_GRACE_FRAMES
        if not acquiring and min(f.pinch_index, f.pinch_middle) < config.PINCH_ON_RATIO:
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
        needed = (
            config.SCROLL_STABLE_FRAMES if raw is Gesture.TWO_FINGERS else config.POSE_STABLE_FRAMES
        )
        if self._pose_frames >= needed:
            self._pose = raw
        return self._pose

    def update(self, landmarks: np.ndarray | None) -> Gesture:
        if landmarks is None:
            self._lost_frames += 1
            if self._lost_frames > config.HAND_LOST_GRACE_FRAMES:
                self.reset()
            return self._last
        self._lost_frames = 0
        self._seen_frames += 1
        f = compute_features(landmarks)
        pose = self._update_pose(f)
        # A fist brings all fingertips close to the thumb, so it must never count as
        # a pinch — checked on the raw current frame, not the debounced pose, so a
        # click cannot slip through while the fist is still being confirmed.
        if f.is_fist:
            self._pinch = self._pinch_candidate = None
            self._pinch_frames = 0
            self._last = pose
            return pose
        pinch = self._update_pinch(f)
        self._last = pinch if pinch is not None else pose
        return self._last


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
                text += (
                    f"  pi={f.pinch_index:.2f} pm={f.pinch_middle:.2f}"
                    f" gap={f.index_middle_gap:.2f}"
                )
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
