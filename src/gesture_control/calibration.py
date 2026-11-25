"""Calibration math: comfortable work area -> cursor sensitivity, hand tremor -> smoothing.

The calibration window (dialogs.py) collects cursor points in two steps:

1. the user sweeps the L pose over the area that is comfortable for the hand;
   the cursor sensitivity is chosen so that this area, moved at a medium
   speed (``CALIB_TARGET_GAIN`` on the acceleration curve), covers the screen;
2. the user holds the hand still; the measured tremor sets the jitter filter
   and the dead zone (more tremor -> stronger smoothing).

All functions are pure, so they are tested without a camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gesture_control import config
from gesture_control.settings import BY_KEY


@dataclass(frozen=True)
class CalibrationResult:
    sensitivity: float
    filter_min_cutoff: float
    deadzone_px: float
    range_x: float        # comfortable range, fraction of the camera frame width
    range_y: float        # ... and height
    jitter_px: float      # tremor while holding still, screen px at base sensitivity

    def as_settings(self) -> dict:
        return {
            "CURSOR_SENSITIVITY": self.sensitivity,
            "FILTER_MIN_CUTOFF": self.filter_min_cutoff,
            "CURSOR_DEADZONE_PX": self.deadzone_px,
        }


def comfort_range(points: np.ndarray) -> tuple[float, float]:
    """Width and height of the swept area, ignoring the outer few percent (outliers)."""
    pts = np.asarray(points, dtype=np.float64)
    lo, hi = config.CALIB_RANGE_PERCENTILES
    rx = float(np.percentile(pts[:, 0], hi) - np.percentile(pts[:, 0], lo))
    ry = float(np.percentile(pts[:, 1], hi) - np.percentile(pts[:, 1], lo))
    return rx, ry


def sensitivity_for(range_x: float, range_y: float) -> float:
    """Sensitivity at which the comfortable range covers the screen at CALIB_TARGET_GAIN.

    In relative mode a hand move of ``dx`` (frame fraction) moves the cursor by
    ``dx / zone_width * sensitivity * gain`` screens; solve for 1 screen in both
    directions and take the larger requirement.
    """
    zone_w = config.WORK_ZONE_RIGHT - config.WORK_ZONE_LEFT
    zone_h = config.WORK_ZONE_BOTTOM - config.WORK_ZONE_TOP
    eps = config.CALIB_MIN_RANGE
    need_x = zone_w / (max(range_x, eps) * config.CALIB_TARGET_GAIN)
    need_y = zone_h / (max(range_y, eps) * config.CALIB_TARGET_GAIN)
    return BY_KEY["CURSOR_SENSITIVITY"].clamp(max(need_x, need_y))


def jitter_px(points: np.ndarray, screen_w: int, screen_h: int) -> float:
    """Tremor of a still hand in screen pixels at base sensitivity (std of the distance)."""
    pts = np.asarray(points, dtype=np.float64)
    zone_w = config.WORK_ZONE_RIGHT - config.WORK_ZONE_LEFT
    zone_h = config.WORK_ZONE_BOTTOM - config.WORK_ZONE_TOP
    px = (pts[:, 0] - pts[:, 0].mean()) / zone_w * screen_w
    py = (pts[:, 1] - pts[:, 1].mean()) / zone_h * screen_h
    return float(np.sqrt(np.mean(px ** 2 + py ** 2)))


def smoothing_for(jitter: float) -> tuple[float, float]:
    """(filter min cutoff, dead zone px) for the measured tremor."""
    cutoff = config.CALIB_CUTOFF_REF / max(jitter, config.CALIB_MIN_JITTER_PX)
    deadzone = jitter * config.ACCEL_MIN_GAIN * config.CALIB_DEADZONE_FACTOR
    return (
        BY_KEY["FILTER_MIN_CUTOFF"].clamp(cutoff),
        BY_KEY["CURSOR_DEADZONE_PX"].clamp(deadzone),
    )


def calibrate(
    sweep: np.ndarray, still: np.ndarray, screen_w: int, screen_h: int
) -> CalibrationResult:
    if len(sweep) < config.CALIB_MIN_SAMPLES or len(still) < config.CALIB_MIN_SAMPLES:
        raise ValueError("not enough hand samples - keep the hand in view and try again")
    rx, ry = comfort_range(sweep)
    jit = jitter_px(still, screen_w, screen_h)
    cutoff, deadzone = smoothing_for(jit)
    return CalibrationResult(sensitivity_for(rx, ry), cutoff, deadzone, rx, ry, jit)
