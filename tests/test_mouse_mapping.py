import pytest

from gesture_control.mouse import CursorMapper, WorkZone

W, H = 1920, 1080
DT = 1 / 30


def test_zone_corners_map_to_screen_corners():
    z = WorkZone(0.2, 0.1, 0.8, 0.7, 1.0)
    assert z.to_screen(0.2, 0.1, W, H) == pytest.approx((0.0, 0.0))
    assert z.to_screen(0.8, 0.7, W, H) == pytest.approx((W - 1, H - 1))
    assert z.to_screen(0.5, 0.4, W, H) == pytest.approx(((W - 1) / 2, (H - 1) / 2))


def test_points_outside_zone_are_clamped():
    z = WorkZone(0.2, 0.1, 0.8, 0.7, 1.0)
    assert z.to_screen(0.0, 0.0, W, H) == (0.0, 0.0)
    assert z.to_screen(1.0, 1.0, W, H) == (W - 1, H - 1)


def test_sensitivity_shrinks_zone_around_center():
    z = WorkZone(0.2, 0.2, 0.8, 0.8, 2.0)
    assert z.effective() == pytest.approx((0.35, 0.35, 0.65, 0.65))
    # point inside the base zone but outside the shrunk one hits the edge
    assert z.to_screen(0.3, 0.5, W, H)[0] == 0.0


def test_mapper_reaches_all_corners_when_holding_still():
    zone = WorkZone(0.2, 0.1, 0.8, 0.7, 1.0)
    corners = {(0.2, 0.1): (0, 0), (0.8, 0.1): (W - 1, 0),
               (0.8, 0.7): (W - 1, H - 1), (0.2, 0.7): (0, H - 1)}
    for (x, y), expected in corners.items():
        m = CursorMapper(W, H, zone)
        out = None
        for i in range(90):
            out = m.update(x, y, i * DT) or out
        assert out == expected


def test_deadzone_suppresses_tiny_moves():
    m = CursorMapper(W, H, WorkZone(0.0, 0.0, 1.0, 1.0, 1.0), deadzone_px=5.0)
    assert m.update(0.5, 0.5, 0.0) is not None
    assert m.update(0.5001, 0.5, DT) is None


def test_filter_resets_after_hand_lost():
    m = CursorMapper(W, H, WorkZone(0.0, 0.0, 1.0, 1.0, 1.0), lost_reset_s=0.3)
    m.update(0.0, 0.0, 0.0)
    # hand reappears far away after a long gap: jump straight there, no gliding
    assert m.update(1.0, 1.0, 1.0) == (W - 1, H - 1)


def test_no_reset_during_continuous_tracking():
    m = CursorMapper(W, H, WorkZone(0.0, 0.0, 1.0, 1.0, 1.0), lost_reset_s=0.3)
    m.update(0.0, 0.0, 0.0)
    x, y = m.update(1.0, 1.0, DT)
    assert x < W - 1 and y < H - 1  # smoothed, not jumped
