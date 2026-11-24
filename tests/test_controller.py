import threading

from gesture_control.controller import ControlState


def test_enabled_by_default():
    assert ControlState().enabled


def test_toggle_flips_and_returns_new_value():
    s = ControlState()
    assert s.toggle() is False
    assert not s.enabled
    assert s.toggle() is True
    assert s.enabled


def test_listeners_notified_on_change_only():
    s = ControlState()
    events = []
    s.subscribe(events.append)
    s.set_enabled(True)   # no change
    s.set_enabled(False)
    s.toggle()
    assert events == [False, True]


def test_toggle_is_thread_safe():
    s = ControlState()
    n = 1000
    threads = [threading.Thread(target=lambda: [s.toggle() for _ in range(n)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # an even total number of toggles brings it back to the start
    assert s.enabled


# -- GestureController ----------------------------------------------------------
import pytest  # noqa: E402

from gesture_control import config  # noqa: E402
from gesture_control.controller import GestureController  # noqa: E402
from gesture_control.gestures import Gesture as G  # noqa: E402
from gesture_control.mouse import CursorMapper, WorkZone  # noqa: E402

W, H = 1000, 1000
DT = 1 / 30
LEFT = G(config.LEFT_BUTTON_GESTURE)
RIGHT = G(config.RIGHT_BUTTON_GESTURE)


class FakeCursor:
    def __init__(self):
        self.events = []
        self.target = None

    def set_target(self, x, y):
        self.target = (x, y)
        self.events.append(("target", (x, y)))

    def jump_to(self, x, y):
        self.target = (x, y)
        self.events.append(("jump", (x, y)))

    def stop_moving(self):
        self.events.append(("stop",))


class FakeButtons:
    def __init__(self):
        self.events = []

    def press(self, b):
        self.events.append(("press", b))

    def release(self, b):
        self.events.append(("release", b))

    def click(self, b):
        self.events.append(("click", b))

    def scroll(self, ticks):
        self.events.append(("scroll", ticks))

    def names(self):
        return [e[0] for e in self.events]


class Rig:
    """Drives a GestureController frame by frame with a simulated clock."""

    def __init__(self):
        self.cursor, self.buttons = FakeCursor(), FakeButtons()
        mapper = CursorMapper(
            W, H, WorkZone(0.0, 0.0, 1.0, 1.0, 1.0), deadzone_px=0.0, mode="absolute"
        )
        self.ctl = GestureController(mapper, self.cursor, self.buttons)
        self.t = 0.0

    def run(self, gesture, point=(0.5, 0.5), seconds=DT):
        for _ in range(max(1, round(seconds / DT))):
            self.ctl.update(gesture, point, self.t)
            self.t += DT


def test_point_moves_cursor():
    r = Rig()
    r.run(G.POINT, (0.2, 0.3), seconds=1.0)
    assert r.cursor.target == pytest.approx((0.2 * (W - 1), 0.3 * (H - 1)), abs=2)
    assert r.buttons.events == []


def test_pinch_is_click_at_pre_pinch_position():
    r = Rig()
    r.run(G.POINT, (0.5, 0.5), seconds=1.0)
    # fingertip drifts while closing the pinch
    r.run(G.POINT, (0.52, 0.52), seconds=DT)
    r.run(LEFT, (0.52, 0.52), seconds=0.1)
    r.run(G.POINT, (0.52, 0.52), seconds=DT)
    assert r.buttons.names() == ["press", "release"]
    jump = next(e for e in r.cursor.events if e[0] == "jump")
    assert jump[1] == pytest.approx((500, 500), abs=3)


def test_cursor_stays_still_after_click_then_resumes():
    r = Rig()
    r.run(G.POINT, (0.5, 0.5), seconds=1.0)
    r.run(LEFT, (0.5, 0.5), seconds=0.1)
    n = len(r.cursor.events)
    r.run(G.POINT, (0.51, 0.5), seconds=config.CLICK_SETTLE_S / 2)  # small drift
    assert not any(e[0] == "target" for e in r.cursor.events[n:])
    r.run(G.POINT, (0.51, 0.5), seconds=config.CLICK_SETTLE_S)
    assert any(e[0] == "target" for e in r.cursor.events[n:])


def test_two_quick_pinches_make_double_click_in_one_spot():
    r = Rig()
    r.run(G.POINT, (0.5, 0.5), seconds=1.0)
    r.run(LEFT, seconds=0.1)
    r.run(G.POINT, seconds=0.1)
    r.run(LEFT, seconds=0.1)
    r.run(G.POINT, seconds=0.1)
    assert r.buttons.names() == ["press", "release", "press", "release"]
    jumps = [e for e in r.cursor.events if e[0] == "jump"]
    assert len(jumps) == 1  # second press reuses the same spot


def test_holding_pinch_and_moving_drags():
    r = Rig()
    r.run(G.POINT, (0.3, 0.3), seconds=1.0)
    r.run(LEFT, (0.3, 0.3), seconds=0.2)
    assert not r.ctl.dragging
    r.run(LEFT, (0.6, 0.6), seconds=0.5)
    assert r.ctl.dragging
    assert r.buttons.names() == ["press"]
    assert r.cursor.target[0] > 0.45 * W
    r.run(G.POINT, (0.6, 0.6))
    assert r.buttons.names() == ["press", "release"]


def test_right_click_pinch():
    r = Rig()
    r.run(G.POINT, seconds=0.5)
    r.run(RIGHT, seconds=0.3)
    r.run(G.POINT)
    assert r.buttons.events == [("click", "right")]


def test_drag_scroll_up_and_down(monkeypatch):
    monkeypatch.setattr(config, "SCROLL_MODE", "drag")
    r = Rig()
    r.run(G.POINT, (0.5, 0.5), seconds=0.5)
    for i in range(10):
        r.run(G.TWO_FINGERS, (0.5, 0.5 - i * 0.01))  # hand moves up
    ups = [e[1] for e in r.buttons.events if e[0] == "scroll"]
    assert ups and all(v > 0 for v in ups)
    # total movement minus the start dead-band (+ one frame of float slack)
    k = config.SCROLL_TICKS_PER_FRAME
    assert (0.09 - config.SCROLL_START_RATIO - 0.01) * k <= sum(ups) <= 0.09 * k
    r.buttons.events.clear()
    for i in range(10):
        r.run(G.TWO_FINGERS, (0.5, 0.41 + i * 0.01))  # hand moves down
    assert all(e[1] < 0 for e in r.buttons.events if e[0] == "scroll")


def test_scroll_freezes_cursor():
    r = Rig()
    r.run(G.POINT, (0.5, 0.5), seconds=0.5)
    n = len(r.cursor.events)
    r.run(G.TWO_FINGERS, (0.5, 0.3), seconds=0.5)
    assert not any(e[0] == "target" for e in r.cursor.events[n:])


def test_fist_hold_toggles_pause():
    r = Rig()
    r.run(G.POINT, seconds=0.5)
    r.run(G.FIST, seconds=config.FIST_HOLD_S + 0.1)
    assert r.ctl.paused
    n = len(r.cursor.events)
    r.run(LEFT, seconds=0.3)
    r.run(G.POINT, (0.1, 0.1), seconds=0.3)
    assert r.buttons.events == [] and not any(e[0] == "target" for e in r.cursor.events[n:])
    r.run(G.FIST, seconds=config.FIST_HOLD_S + 0.1)
    assert not r.ctl.paused


def test_long_fist_toggles_only_once():
    r = Rig()
    r.run(G.FIST, seconds=config.FIST_HOLD_S * 3)
    assert r.ctl.paused


def test_short_fist_does_not_pause():
    r = Rig()
    r.run(G.FIST, seconds=config.FIST_HOLD_S / 2)
    r.run(G.POINT)
    assert not r.ctl.paused


def test_lost_hand_releases_held_button():
    r = Rig()
    r.run(G.POINT, seconds=0.5)
    r.run(LEFT, seconds=0.2)
    r.run(LEFT, point=None, seconds=config.HAND_LOST_RELEASE_S / 2)
    assert r.buttons.names() == ["press"]  # brief dropout keeps the drag
    r.run(LEFT, point=None, seconds=config.HAND_LOST_RELEASE_S)
    assert r.buttons.names() == ["press", "release"]


def test_release_all_lets_go_of_button():
    r = Rig()
    r.run(G.POINT, seconds=0.5)
    r.run(LEFT, seconds=0.2)
    r.ctl.release_all()
    assert r.buttons.names() == ["press", "release"]
    r.ctl.release_all()
    assert r.buttons.names() == ["press", "release"]


def test_default_mapping_left_is_index_right_is_middle():
    assert LEFT is G.PINCH_INDEX
    assert RIGHT is G.PINCH_MIDDLE


def test_quick_pinch_with_hand_moving_is_click_not_drag():
    """Hand drifts during a short pinch: must not turn into a selection drag."""
    r = Rig()
    r.run(G.POINT, (0.5, 0.5), seconds=1.0)
    for i in range(int(config.DRAG_MIN_HOLD_S / DT) - 1):
        r.run(LEFT, (0.5 + i * 0.02, 0.5))
    assert not r.ctl.dragging
    r.run(G.POINT, (0.6, 0.5))
    assert r.buttons.names() == ["press", "release"]


def test_jitter_during_long_pinch_does_not_start_drag():
    r = Rig()
    r.run(G.POINT, (0.5, 0.5), seconds=1.0)
    for i in range(30):  # 1 s of pinch with +-2% raw jitter
        r.run(LEFT, (0.5 + (0.02 if i % 2 else -0.02), 0.5))
    assert not r.ctl.dragging


@pytest.mark.parametrize("mode", ["auto", "drag"])
def test_entering_scroll_pose_does_not_nudge_page(monkeypatch, mode):
    monkeypatch.setattr(config, "SCROLL_MODE", mode)
    r = Rig()
    r.run(G.POINT, (0.5, 0.5), seconds=0.5)
    r.run(G.TWO_FINGERS, (0.5, 0.5 - config.SCROLL_START_RATIO / 2), seconds=0.5)
    assert not any(e[0] == "scroll" for e in r.buttons.events)



# -- auto scroll (mouse-wheel click style) --------------------------------------
def scrolled(r):
    return sum(e[1] for e in r.buttons.events if e[0] == "scroll")


def auto_rig(monkeypatch):
    monkeypatch.setattr(config, "SCROLL_MODE", "auto")
    r = Rig()
    r.run(G.POINT, (0.5, 0.5), seconds=0.3)
    r.run(G.TWO_FINGERS, (0.5, 0.5))  # anchor at y = 0.5
    return r


def test_auto_scroll_keeps_scrolling_while_hand_is_held_above_anchor(monkeypatch):
    r = auto_rig(monkeypatch)
    offset = config.SCROLL_AUTO_DEADZONE + 0.1
    r.run(G.TWO_FINGERS, (0.5, 0.5 - offset), seconds=1.0)
    first = scrolled(r)
    r.run(G.TWO_FINGERS, (0.5, 0.5 - offset), seconds=1.0)  # hand not moving
    second = scrolled(r) - first
    expected = 0.1 * config.SCROLL_AUTO_SPEED  # ticks per second
    assert first > 0 and second == pytest.approx(expected, rel=0.1)


def test_auto_scroll_down_below_anchor(monkeypatch):
    r = auto_rig(monkeypatch)
    r.run(G.TWO_FINGERS, (0.5, 0.5 + config.SCROLL_AUTO_DEADZONE + 0.1), seconds=1.0)
    assert scrolled(r) < 0


def test_auto_scroll_faster_further_from_anchor(monkeypatch):
    near, far = auto_rig(monkeypatch), auto_rig(monkeypatch)
    d = config.SCROLL_AUTO_DEADZONE
    near.run(G.TWO_FINGERS, (0.5, 0.5 - d - 0.05), seconds=1.0)
    far.run(G.TWO_FINGERS, (0.5, 0.5 - d - 0.15), seconds=1.0)
    assert scrolled(far) > scrolled(near) > 0


def test_auto_scroll_dead_zone_ignores_small_offsets(monkeypatch):
    r = auto_rig(monkeypatch)
    r.run(G.TWO_FINGERS, (0.5, 0.5 - config.SCROLL_AUTO_DEADZONE * 0.8), seconds=2.0)
    assert scrolled(r) == 0


def test_auto_scroll_speed_is_capped(monkeypatch):
    r = auto_rig(monkeypatch)
    r.run(G.TWO_FINGERS, (0.5, 0.0), seconds=1.0)
    assert scrolled(r) <= config.SCROLL_AUTO_MAX_TICKS_S * 1.05


def test_auto_scroll_stops_when_fingers_open(monkeypatch):
    r = auto_rig(monkeypatch)
    r.run(G.TWO_FINGERS, (0.5, 0.3), seconds=0.5)
    before = len(r.buttons.events)
    r.run(G.POINT, (0.5, 0.3), seconds=1.0)
    assert len(r.buttons.events) == before
    assert r.ctl.scroll_anchor is None


def test_auto_scroll_anchor_resets_on_new_pose(monkeypatch):
    r = auto_rig(monkeypatch)
    r.run(G.TWO_FINGERS, (0.5, 0.3), seconds=0.3)
    r.run(G.POINT, (0.5, 0.3), seconds=0.3)
    r.run(G.TWO_FINGERS, (0.5, 0.3))  # new anchor where the hand is now
    assert r.ctl.scroll_anchor == pytest.approx(0.3)
    before = scrolled(r)
    r.run(G.TWO_FINGERS, (0.5, 0.3), seconds=1.0)
    assert scrolled(r) == before


def test_auto_scroll_invert(monkeypatch):
    monkeypatch.setattr(config, "SCROLL_INVERT", True)
    r = auto_rig(monkeypatch)
    r.run(G.TWO_FINGERS, (0.5, 0.3), seconds=0.5)
    assert scrolled(r) < 0


# -- detector + controller together ---------------------------------------------------
def run_detector_and_controller(frames):
    from gesture_control.gestures import GestureDetector

    r = Rig()
    d = GestureDetector()
    for lm in frames:
        r.ctl.update(d.update(lm), (0.5, 0.5), r.t)
        r.t += DT
    return r


def test_quick_index_pinch_tap_is_left_click_end_to_end():
    from handgen import make_hand

    point = make_hand(("index",))
    pinch = make_hand(("index",), thumb_tip="index")
    r = run_detector_and_controller([point] * 15 + [pinch] * 2 + [point] * 10)
    assert r.buttons.events == [("press", "left"), ("release", "left")]


def test_middle_pinch_is_right_click_end_to_end():
    from handgen import make_hand

    point = make_hand(("index",))
    pinch = make_hand(("index", "middle"), thumb_tip="middle")
    r = run_detector_and_controller([point] * 15 + [pinch] * 8 + [point] * 10)
    assert r.buttons.events == [("click", "right")]


def test_brief_middle_pinch_is_not_right_click_end_to_end():
    from handgen import make_hand

    point = make_hand(("index",))
    pinch = make_hand(("index", "middle"), thumb_tip="middle")
    r = run_detector_and_controller([point] * 15 + [pinch] * 3 + [point] * 10)
    assert r.buttons.events == []


def test_controller_reports_last_event():
    r = Rig()
    r.run(G.POINT, seconds=0.5)
    r.run(LEFT, seconds=0.1)
    assert r.ctl.last_event[0] == "LEFT DOWN"
    r.run(G.POINT)
    assert r.ctl.last_event[0] == "LEFT CLICK"
    r.run(G.POINT, seconds=0.5)
    r.run(RIGHT, seconds=0.3)
    assert r.ctl.last_event[0] == "RIGHT CLICK"



def test_right_click_waits_for_minimum_hold():
    r = Rig()
    r.run(G.POINT, seconds=0.5)
    r.run(RIGHT, seconds=config.RIGHT_CLICK_MIN_HOLD_S / 2)
    assert r.buttons.events == []
    r.run(RIGHT, seconds=config.RIGHT_CLICK_MIN_HOLD_S)
    assert r.buttons.events == [("click", "right")]
    r.run(RIGHT, seconds=1.0)  # holding longer does not repeat it
    assert r.buttons.events == [("click", "right")]


def test_left_button_held_still_is_auto_released():
    r = Rig()
    r.run(G.POINT, (0.5, 0.5), seconds=0.5)
    r.run(LEFT, (0.5, 0.5), seconds=config.LEFT_HOLD_STILL_MAX_S + 0.2)
    assert r.buttons.names() == ["press", "release"]
    assert r.ctl.last_event[0] == "AUTO RELEASE"
    r.run(G.POINT)  # opening the fingers afterwards does not release twice
    assert r.buttons.names() == ["press", "release"]


def test_dragging_is_never_auto_released():
    r = Rig()
    r.run(G.POINT, (0.3, 0.3), seconds=0.5)
    r.run(LEFT, (0.3, 0.3), seconds=0.4)
    r.run(LEFT, (0.6, 0.6), seconds=config.LEFT_HOLD_STILL_MAX_S + 0.5)
    assert r.ctl.dragging and r.buttons.names() == ["press"]
