import numpy as np
import pytest

from gesture_control import config
from gesture_control.calibration import (
    calibrate, comfort_range, jitter_px, sensitivity_for, smoothing_for,
)
from gesture_control.settings import BY_KEY

W, H = 2560, 1440


def sweep(rx, ry, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([0.5 + rng.uniform(-rx / 2, rx / 2, n),
                            0.4 + rng.uniform(-ry / 2, ry / 2, n)])


def still(std, n=90, seed=1):
    rng = np.random.default_rng(seed)
    return np.column_stack([0.5 + rng.normal(0, std, n), 0.4 + rng.normal(0, std, n)])


def test_comfort_range_ignores_outliers():
    pts = sweep(0.3, 0.2)
    pts[0] = (5.0, 5.0)  # one tracking glitch
    rx, ry = comfort_range(pts)
    assert rx == pytest.approx(0.3 * 0.9, rel=0.1) and ry == pytest.approx(0.2 * 0.9, rel=0.1)


def test_smaller_comfort_area_gives_higher_sensitivity():
    assert sensitivity_for(0.15, 0.1) > sensitivity_for(0.4, 0.3)


def test_sensitivity_makes_comfort_area_cover_the_screen():
    s = sensitivity_for(0.3, 0.25)
    zone_w = config.WORK_ZONE_RIGHT - config.WORK_ZONE_LEFT
    zone_h = config.WORK_ZONE_BOTTOM - config.WORK_ZONE_TOP
    screens_x = 0.3 / zone_w * s * config.CALIB_TARGET_GAIN
    screens_y = 0.25 / zone_h * s * config.CALIB_TARGET_GAIN
    assert min(screens_x, screens_y) == pytest.approx(1.0, rel=1e-6)


def test_sensitivity_is_clamped():
    s = BY_KEY["CURSOR_SENSITIVITY"]
    assert sensitivity_for(0.001, 0.001) == s.maximum
    assert sensitivity_for(5.0, 5.0) == s.minimum


def test_jitter_grows_with_tremor():
    assert jitter_px(still(0.0005), W, H) < jitter_px(still(0.003), W, H)
    assert jitter_px(np.tile([0.5, 0.4], (50, 1)), W, H) == pytest.approx(0.0, abs=1e-6)


def test_more_tremor_gives_stronger_smoothing():
    calm_cutoff, calm_dz = smoothing_for(1.0)
    shaky_cutoff, shaky_dz = smoothing_for(8.0)
    assert shaky_cutoff < calm_cutoff and shaky_dz > calm_dz


def test_calibrate_end_to_end_and_settings():
    r = calibrate(sweep(0.3, 0.2), still(0.001), W, H)
    applied = r.as_settings()
    for key, value in applied.items():
        s = BY_KEY[key]
        assert s.minimum <= value <= s.maximum
    assert r.jitter_px > 0


def test_calibrate_needs_enough_samples():
    with pytest.raises(ValueError):
        calibrate(sweep(0.3, 0.2, n=5), still(0.001), W, H)
