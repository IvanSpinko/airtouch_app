import random
import statistics

from gesture_control.filters import OneEuroFilter, PointFilter

RATE = 30.0


def run(f, values):
    return [f(v, i / RATE) for i, v in enumerate(values)]


def test_first_value_passes_through():
    f = OneEuroFilter()
    assert f(123.0, 0.0) == 123.0


def test_constant_signal_stays_constant():
    out = run(OneEuroFilter(), [42.0] * 50)
    assert all(abs(v - 42.0) < 1e-9 for v in out)


def test_reduces_jitter_at_rest():
    rng = random.Random(0)
    raw = [500.0 + rng.gauss(0.0, 5.0) for _ in range(300)]
    out = run(OneEuroFilter(), raw)
    # skip the filter warm-up
    assert statistics.pstdev(out[30:]) < statistics.pstdev(raw[30:]) / 2


def test_fast_motion_has_less_lag_with_higher_beta():
    ramp = [i * 50.0 for i in range(60)]  # 1500 px/s
    slow = run(OneEuroFilter(beta=0.0), ramp)
    fast = run(OneEuroFilter(beta=0.05), ramp)
    assert abs(ramp[-1] - fast[-1]) < abs(ramp[-1] - slow[-1])


def test_converges_to_step():
    out = run(OneEuroFilter(), [0.0] * 5 + [100.0] * 90)
    assert abs(out[-1] - 100.0) < 1.0


def test_non_increasing_time_returns_previous():
    f = OneEuroFilter()
    f(10.0, 1.0)
    assert f(50.0, 1.0) == 10.0
    assert f(50.0, 0.5) == 10.0


def test_reset_forgets_state():
    f = OneEuroFilter()
    f(0.0, 0.0)
    f(0.0, 0.1)
    f.reset()
    assert f(999.0, 0.2) == 999.0


def test_point_filter_filters_both_axes():
    pf = PointFilter()
    assert pf(1.0, 2.0, 0.0) == (1.0, 2.0)
    x, y = pf(11.0, 12.0, 1 / RATE)
    assert 1.0 < x < 11.0 and 2.0 < y < 12.0
