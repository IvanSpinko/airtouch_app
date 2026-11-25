"""Gesture recognition from the 21 hand landmarks.

Gesture set (stage 4 redesign, chosen to be easy for MediaPipe to tell apart - big
finger movements instead of fingertip contact):

- ``l_pose``      index + thumb straight, other fingers folded ("L") - moves the cursor
- ``thumb_bent``  from L, the thumb bends in - left click / drag
- ``index_bent``  from L, the index finger bends - right click
- ``two_fingers`` index + middle together, thumb tucked - scroll
- ``fist``        pause
- ``shaka``       only thumb + pinky out - on-screen keyboard
- ``open_palm``   resting hand (does nothing)
- ``none``        anything else

Rules work without a model; a trained classifier (stage 3) takes over when present.
All measurements are 3D and normalized by palm size, and left hands are mirrored
onto right ones, so distance to the camera and the hand used do not matter.

Check: ``python -m gesture_control.gestures`` shows the recognized gesture live.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from gesture_control import config


class Gesture(str, Enum):
    NONE = "none"
    L_POSE = "l_pose"
    THUMB_BENT = "thumb_bent"
    INDEX_BENT = "index_bent"
    TWO_FINGERS = "two_fingers"
    FIST = "fist"
    OPEN_PALM = "open_palm"
    SHAKA = "shaka"


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
    # Thumb tip distance from the middle-finger knuckle: large when the thumb sticks
    # out (L pose), small when it is bent in across the palm.
    thumb_reach: float = 0.0

    @property
    def extended_count(self) -> int:
        return sum(self.extended.values())

    @property
    def thumb_out(self) -> bool:
        return self.thumb_reach > config.THUMB_EXTENDED_RATIO

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
        thumb_reach=dist(config.LM_THUMB_TIP, config.LM_MIDDLE_MCP) / palm,
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
        if not (config.USE_CLASSIFIER and path.exists()):
            return None
        classes = tuple(str(c) for c in np.load(path, allow_pickle=False)["classes"])
        if classes != tuple(config.GESTURE_CLASSES):
            print(f"Ignoring {path.name}: it was trained for another gesture set. "
                  "Record and retrain: python -m ml.record, python -m ml.train")
            return None
        return cls(path)

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
    """Pose from finger states (rules, no history). Used when there is no model."""
    e = f.extended
    folded3 = not (e["middle"] or e["ring"] or e["pinky"])
    if f.is_fist and not f.thumb_out:
        return Gesture.FIST
    if f.thumb_out and e["pinky"] and not (e["index"] or e["middle"] or e["ring"]):
        return Gesture.SHAKA
    if f.extended_count == 4 and f.thumb_out:
        return Gesture.OPEN_PALM
    if (
        e["index"] and e["middle"] and not e["ring"] and not e["pinky"]
        # Fingers together and thumb away from both tips.
        and f.index_middle_gap < config.SCROLL_FINGERS_TOGETHER_RATIO
        and min(f.pinch_index, f.pinch_middle) > config.SCROLL_THUMB_AWAY_RATIO
    ):
        return Gesture.TWO_FINGERS
    if e["index"] and folded3:
        return Gesture.L_POSE if f.thumb_out else Gesture.THUMB_BENT
    if not e["index"] and folded3 and f.thumb_out:
        return Gesture.INDEX_BENT
    return Gesture.NONE


CLICK_POSES = (Gesture.THUMB_BENT, Gesture.INDEX_BENT)


class GestureDetector:
    """Stateful detector: smoothed model probabilities + debounced poses.

    - With a trained ``classifier`` the pose is the model's top class (averaged
      over the last frames, ``CLASSIFIER_SMOOTHING``) when it is confident, else
      the rules decide.
    - A pose must be seen for a few frames in a row before it counts: click poses
      ``CLICK_POSE_FRAMES``, scroll ``SCROLL_STABLE_FRAMES``, others
      ``POSE_STABLE_FRAMES``.
    - Right after the hand appears its landmarks are unreliable, so click poses
      are ignored for ``ACQUIRE_GRACE_FRAMES``.
    """

    def __init__(self, classifier: GestureClassifier | None = None) -> None:
        self.classifier = classifier
        self.probs: dict[Gesture, float] | None = None
        self.features: HandFeatures | None = None
        # How strongly each click pose is shown (0..1), for the preview bars.
        self.strength: dict[Gesture, float] = {g: 0.0 for g in CLICK_POSES}
        self.reset()

    def reset(self) -> None:
        self._lost_frames = 0
        self._seen_frames = 0
        self._last = Gesture.NONE
        self._smoothed: dict[Gesture, float] | None = None
        self._pose = Gesture.NONE
        self._pose_candidate = Gesture.NONE
        self._pose_frames = 0

    def _smooth(self, probs: dict[Gesture, float]) -> dict[Gesture, float]:
        """Exponential average over frames: one odd frame cannot flip the pose."""
        a = config.CLASSIFIER_SMOOTHING
        if self._smoothed is None:
            self._smoothed = dict(probs)
        else:
            self._smoothed = {
                g: a * p + (1 - a) * self._smoothed.get(g, p) for g, p in probs.items()
            }
        return self._smoothed

    def _raw_pose(self, f: HandFeatures) -> Gesture:
        if self.probs is not None:
            top = max(self.probs, key=self.probs.get)
            if self.probs[top] >= config.CLASSIFIER_MIN_PROB:
                return top
        return classify_pose(f)

    def _update_pose(self, raw: Gesture) -> Gesture:
        if raw is self._pose_candidate:
            self._pose_frames += 1
        else:
            self._pose_candidate = raw
            self._pose_frames = 1
        if raw is Gesture.TWO_FINGERS:
            needed = config.SCROLL_STABLE_FRAMES
        elif raw in CLICK_POSES:
            needed = config.CLICK_POSE_FRAMES
        else:
            needed = config.POSE_STABLE_FRAMES
        if self._pose_frames >= needed:
            self._pose = raw
        return self._pose

    def update(self, landmarks: np.ndarray | None, handedness: str = "Right") -> Gesture:
        if landmarks is None:
            self._lost_frames += 1
            if self._lost_frames > config.HAND_LOST_GRACE_FRAMES:
                self.reset()
            return self._last
        self._lost_frames = 0
        self._seen_frames += 1
        f = compute_features(landmarks)
        self.features = f
        self.probs = (
            self._smooth(self.classifier.predict(landmarks, handedness))
            if self.classifier else None
        )
        raw = self._raw_pose(f)
        if self.probs is not None:
            self.strength = {g: self.probs.get(g, 0.0) for g in CLICK_POSES}
        else:
            self.strength = {g: float(raw is g) for g in CLICK_POSES}
        if raw in CLICK_POSES and self._seen_frames <= config.ACQUIRE_GRACE_FRAMES:
            raw = Gesture.NONE
        self._last = self._update_pose(raw)
        return self._last


def _demo() -> None:
    import cv2

    from gesture_control.camera import Camera
    from gesture_control.tracker import HandTracker, draw_hand

    detector = GestureDetector(GestureClassifier.load_if_available())
    print("classifier:", "trained model" if detector.classifier else "rules only")
    with Camera() as cam, HandTracker() as tracker:
        while cam.is_running:
            item = cam.read(timeout=1.0)
            if item is None:
                continue
            frame, ts = item
            hand = tracker.process(frame, ts)
            gesture = detector.update(hand.landmarks if hand else None,
                                      hand.handedness if hand else "Right")
            frame = frame.copy()
            text = gesture.value
            if hand is not None:
                draw_hand(frame, hand)
                f = compute_features(hand.landmarks)
                text += f"  thumb={f.thumb_reach:.2f} gap={f.index_middle_gap:.2f}"
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
