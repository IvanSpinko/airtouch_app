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
        m = CursorMapper(W, H, zone, mode="absolute")
        out = None
        for i in range(90):
            out = m.update(x, y, i * DT) or out
        assert out == expected


def test_deadzone_suppresses_tiny_moves():
    m = CursorMapper(W, H, WorkZone(0.0, 0.0, 1.0, 1.0, 1.0), deadzone_px=5.0)
    assert m.update(0.5, 0.5, 0.0) is not None
    assert m.update(0.5001, 0.5, DT) is None


def test_filter_resets_after_hand_lost():
    m = CursorMapper(W, H, WorkZone(0.0, 0.0, 1.0, 1.0, 1.0), lost_reset_s=0.3, mode="absolute")
    m.update(0.0, 0.0, 0.0)
    # hand reappears far away after a long gap: jump straight there, no gliding
    assert m.update(1.0, 1.0, 1.0) == (W - 1, H - 1)


def test_no_reset_during_continuous_tracking():
    m = CursorMapper(W, H, WorkZone(0.0, 0.0, 1.0, 1.0, 1.0), lost_reset_s=0.3, mode="absolute")
    m.update(0.5, 0.5, 0.0)
    x, y = m.update(0.55, 0.55, DT)  # fast but plausible hand move
    assert 0.5 * (W - 1) < x < 0.55 * (W - 1)  # smoothed, not jumped


def test_glide_moves_part_of_the_way():
    from gesture_control.mouse import glide_step

    x, y = glide_step((0.0, 0.0), (100.0, 0.0), dt=0.01, tau=0.025)
    assert 0.0 < x < 100.0 and y == 0.0


def test_glide_converges_and_snaps():
    from gesture_control.mouse import glide_step

    p = (0.0, 0.0)
    for _ in range(200):
        p = glide_step(p, (100.0, 50.0), dt=1 / 144, tau=0.025)
    assert p == (100.0, 50.0)



# -- relative mode with acceleration --------------------------------------------
from gesture_control import config  # noqa: E402
from gesture_control.mouse import accel_gain  # noqa: E402

FULL = WorkZone(0.0, 0.0, 1.0, 1.0, 1.0)


def relative(**kw):
    kw.setdefault("deadzone_px", 0.0)
    return CursorMapper(W, H, FULL, mode="relative", position_source=lambda: (500, 500), **kw)


def travel(mapper, x0, x1, seconds):
    """Moves the hand from x0 to x1 over ``seconds``; returns cursor x displacement."""
    frames = max(2, round(seconds / DT))
    t0 = mapper._last_seen + 1.0 if mapper._last_seen is not None else 0.0
    start = None
    out = None
    for i in range(frames + 30):  # extra frames let the filter settle at x1
        k = min(i / frames, 1.0)
        out = mapper.update(x0 + (x1 - x0) * k, 0.5, t0 + i * DT) or out
        start = start or out
    return out[0] - start[0]


def test_accel_gain_curve():
    assert accel_gain(0.0) == config.ACCEL_MIN_GAIN
    assert accel_gain(1e9) == config.ACCEL_MAX_GAIN
    mid = accel_gain((config.ACCEL_LOW_SPEED_PX_S + config.ACCEL_HIGH_SPEED_PX_S) / 2)
    assert config.ACCEL_MIN_GAIN < mid < config.ACCEL_MAX_GAIN
    speeds = [i * 100.0 for i in range(60)]
    gains = [accel_gain(v) for v in speeds]
    assert gains == sorted(gains)  # monotonic


def test_slow_movement_is_more_precise_than_fast():
    slow = travel(relative(), 0.45, 0.55, seconds=2.0)
    fast = travel(relative(), 0.45, 0.55, seconds=0.1)
    assert 0 < slow < fast
    # slow aiming: well below the 1:1 absolute gain (0.1 of the screen width)
    assert slow < 0.1 * W * 0.6


def test_relative_starts_from_real_cursor_and_does_not_jump():
    m = relative()
    assert m.update(0.9, 0.9, 0.0) == (500, 500)  # first frame: no jump to the hand


def test_relative_hand_reappearing_elsewhere_does_not_jump():
    m = relative()
    m.update(0.2, 0.2, 0.0)
    m.update(0.2, 0.2, DT)
    pos = m.update(0.8, 0.8, 5.0)  # hand lost, then shown in a different place
    assert pos == (500, 500)


def test_relative_cursor_is_clamped_to_screen():
    m = relative()
    out = None
    for i in range(60):  # big fast sweeps to the right
        out = m.update(i * 0.05, 0.5, i * DT) or out
    assert out[0] == W - 1


def test_relative_reaches_corner_with_a_flick():
    m = relative()
    out = None
    for i in range(10):
        out = m.update(0.5 - i * 0.06, 0.5 - i * 0.06, i * DT) or out
    assert out == (0, 0)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        CursorMapper(W, H, mode="nope")



# -- outlier rejection ------------------------------------------------------------
def test_single_frame_jump_is_ignored():
    m = relative()
    for i in range(10):
        m.update(0.5, 0.5, i * DT)
    pos = m.update(0.5, 0.5, 10 * DT) or (500, 500)
    assert m.update(0.95, 0.1, 11 * DT) is None  # glitch frame
    after = m.update(0.5, 0.5, 12 * DT)
    assert after is None or after == pos


def test_persistent_jump_reanchors_without_moving_cursor():
    m = relative()
    for i in range(10):
        m.update(0.5, 0.5, i * DT)
    outs = [m.update(0.9, 0.9, (10 + k) * DT) for k in range(config.OUTLIER_CONFIRM_FRAMES + 3)]
    assert all(o is None or o == (500, 500) for o in outs)
    # and normal tracking resumes from the new place
    moved = None
    for k in range(20):
        moved = m.update(0.9 + k * 0.005, 0.9, (20 + k) * DT) or moved
    assert moved is not None and moved[0] > 500


def test_absolute_mode_also_rejects_glitch():
    m = CursorMapper(W, H, FULL, deadzone_px=0.0, mode="absolute")
    for i in range(30):
        m.update(0.5, 0.5, i * DT)
    assert m.update(0.0, 0.0, 30 * DT) is None


def test_accel_gain_reads_config_at_call_time(monkeypatch):
    monkeypatch.setattr(config, "ACCEL_MAX_GAIN", 9.0)
    assert accel_gain(1e9) == 9.0
