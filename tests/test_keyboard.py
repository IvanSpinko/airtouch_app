"""On-screen keyboard helpers that do not touch Windows."""

import pytest

from gesture_control import config
from gesture_control.keyboard import enlarged_geometry


@pytest.mark.parametrize("w,h,scale", [(2560, 1440, 1.25), (1920, 1080, 1.0), (3840, 2160, 1.5)])
def test_enlarged_geometry_fits_screen_and_is_centered(w, h, scale):
    g = enlarged_geometry(w, h, scale)
    w_dip, h_dip = w / scale, h / scale
    assert g.width == pytest.approx(w_dip * config.OSK_WIDTH_FRACTION, abs=1)
    assert g.height == pytest.approx(h_dip * config.OSK_HEIGHT_FRACTION, abs=1)
    assert 0 <= g.left and g.left + g.width <= w_dip + 1
    assert 0 <= g.top and g.top + g.height <= h_dip + 1
    assert abs(g.left - (w_dip - g.left - g.width)) <= 1  # centered horizontally
    assert g.top >= h_dip / 2  # in the lower half


def test_geometry_is_larger_than_osk_default():
    g = enlarged_geometry(2560, 1440, 1.25)
    # osk.exe default is 1638 x 483: at least 10% bigger in both directions
    assert g.width > 0x666 * 1.1 and g.height > 0x1E3 * 1.1
