"""Gesture dataset: CSV storage, feature extraction, splits and augmentation.

CSV layout (``data/gestures.csv``): ``label, handedness, session, x0, y0, z0, ..., x20, y20, z20``.
Only hand landmarks are stored — never camera images.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gesture_control import config
from gesture_control.gestures import normalize_landmarks

COORDS = [f"{axis}{i}" for i in range(config.NUM_LANDMARKS) for axis in "xyz"]
HEADER = ["label", "handedness", "session", *COORDS]


def append_rows(path: Path, rows: list[tuple[str, str, str, np.ndarray]]) -> None:
    """Appends (label, handedness, session, landmarks[21, 3]) rows, writing a header if new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        for label, hand, session, lm in rows:
            w.writerow([label, hand, session, *(f"{v:.5f}" for v in np.asarray(lm).reshape(-1))])


def remove_last_rows(path: Path, count: int) -> int:
    """Deletes the last ``count`` data rows (undo while recording). Returns rows removed."""
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    removable = max(len(lines) - 1, 0)
    n = min(count, removable)
    if n:
        path.write_text("\n".join(lines[: len(lines) - n]) + "\n", encoding="utf-8")
    return n


@dataclass
class Dataset:
    landmarks: np.ndarray   # (N, 21, 3) raw normalized-frame coordinates
    handedness: np.ndarray  # (N,) "Left" / "Right"
    labels: np.ndarray      # (N,) int class index into config.GESTURE_CLASSES
    sessions: np.ndarray    # (N,) recording session id

    def __len__(self) -> int:
        return len(self.labels)

    def features(self) -> np.ndarray:
        return np.stack([
            normalize_landmarks(lm, h) for lm, h in zip(self.landmarks, self.handedness)
        ]) if len(self) else np.zeros((0, config.NUM_LANDMARKS * 3), np.float32)

    def counts(self) -> dict[str, int]:
        return {
            name: int((self.labels == i).sum()) for i, name in enumerate(config.GESTURE_CLASSES)
        }


def load(path: Path = config.GESTURE_DATA_PATH) -> Dataset:
    lms, hands, labels, sessions = [], [], [], []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["label"] not in config.GESTURE_CLASSES:
                    continue
                labels.append(config.GESTURE_CLASSES.index(row["label"]))
                hands.append(row["handedness"])
                sessions.append(row["session"])
                lms.append([float(row[c]) for c in COORDS])
    return Dataset(
        landmarks=np.asarray(lms, np.float32).reshape(-1, config.NUM_LANDMARKS, 3),
        handedness=np.asarray(hands),
        labels=np.asarray(labels, np.int64),
        sessions=np.asarray(sessions),
    )


def stratified_split(
    labels: np.ndarray,
    val: float = config.ML_VAL_FRACTION,
    test: float = config.ML_TEST_FRACTION,
    seed: int = config.ML_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / val / test indices with the same class proportions in each part."""
    rng = np.random.default_rng(seed)
    parts: tuple[list, list, list] = ([], [], [])
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_test = int(round(len(idx) * test))
        n_val = int(round(len(idx) * val))
        parts[2].extend(idx[:n_test])
        parts[1].extend(idx[n_test:n_test + n_val])
        parts[0].extend(idx[n_test + n_val:])
    return tuple(np.sort(np.asarray(p, np.int64)) for p in parts)  # type: ignore[return-value]


def augment(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random in-plane rotation, scale and noise on normalized feature vectors."""
    pts = features.reshape(len(features), -1, 3).copy()
    ang = np.deg2rad(rng.uniform(-config.ML_AUG_ROTATION_DEG, config.ML_AUG_ROTATION_DEG, len(pts)))
    cos, sin = np.cos(ang)[:, None], np.sin(ang)[:, None]
    x, y = pts[:, :, 0].copy(), pts[:, :, 1].copy()
    pts[:, :, 0] = cos * x - sin * y
    pts[:, :, 1] = sin * x + cos * y
    pts *= rng.uniform(1 - config.ML_AUG_SCALE, 1 + config.ML_AUG_SCALE, (len(pts), 1, 1))
    pts += rng.normal(0.0, config.ML_AUG_NOISE, pts.shape)
    return pts.reshape(len(features), -1).astype(np.float32)
