"""Optional session log for diagnosing false clicks.

``python -m gesture_control.main --log`` writes one row per frame to
``data/session-<time>.csv``: time, hand landmarks (no images), recognized gesture,
pinch strengths and the mouse action of that frame. Pressing ``m`` in the preview
adds a MARK row ("something went wrong here"). ``python -m ml.replay`` replays the
log through the current code.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from gesture_control import config

COORDS = [f"{axis}{i}" for i in range(config.NUM_LANDMARKS) for axis in "xyz"]
HEADER = ["t", "handedness", "gesture", "strength_left", "strength_right", "event", *COORDS]


class SessionLogger:
    def __init__(self, directory: Path = config.DATA_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / time.strftime(config.SESSION_LOG_NAME)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(HEADER)
        self._last_event: tuple[str, float] | None = None

    def write(
        self, t: float, landmarks: np.ndarray | None, handedness: str, gesture: str,
        strength_left: float, strength_right: float, event: tuple[str, float] | None,
    ) -> None:
        new_event = event[0] if event is not None and event != self._last_event else ""
        self._last_event = event
        coords = (
            [f"{v:.5f}" for v in np.asarray(landmarks).reshape(-1)] if landmarks is not None
            else [""] * len(COORDS)
        )
        self._writer.writerow([
            f"{t:.4f}", handedness if landmarks is not None else "", gesture,
            f"{strength_left:.3f}", f"{strength_right:.3f}", new_event, *coords,
        ])

    def mark(self, t: float) -> None:
        self._writer.writerow([f"{t:.4f}", "", "", "", "", "MARK", *[""] * len(COORDS)])
        self._file.flush()

    def close(self) -> None:
        self._file.close()
