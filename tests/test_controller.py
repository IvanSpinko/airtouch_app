import threading

import pytest
from handgen import pose

from gesture_control import config
from gesture_control.controller import ControlState, GestureController
from gesture_control.gestures import Gesture as G
from gesture_control.gestures import GestureDetector
from gesture_control.mouse import CursorMapper, WorkZone


# -- ControlState --------------------------------------------------------------------
def test_enabled_by_default():
    assert ControlState().enabled


def test_toggle_flips_and_returns_new_value():
    s = ControlState()
    assert s.toggle() is False and not s.enabled
    assert s.toggle() is True and s.enabled


def test_listeners_notified_on_change_only():
    s = ControlState()
    events = []
    s.subscribe(events.append)
    s.set_enabled(True)  # no change
    s.set_enabled(False)
    s.toggle()
    assert events == [False, True]


def test_toggle_is_thread_safe():
    s = ControlState()
    threads = [threading.Thread(target=lambda: [s.toggle() for _ in range(1000)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert s.enabled  # an even number of toggles


# -- GestureController ---------------------------------------------------------------
W, H = 1000, 1000
DT = 1 / 30
L = G.L_POSE
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

    def moved(self, since=0):
        return any(e[0] == "target" for e in self.events[since:])


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


class Rig:
    """Drives a GestureController frame by frame with a simulated clock."""

    def __init__(self):
        self.cursor, self.buttons = FakeCursor(), FakeButtons()
        mapper = CursorMapper(
            W, H, WorkZone(0.0, 0.0, 1.0, 1.0, 1.0), deadzone_px=0.0, mode="absolute"
        )
        self.keyboard_calls = []
        self.ctl = GestureController(
            mapper, self.cursor, self.buttons,
            on_keyboard_toggle=lambda: self.keyboard_calls.append(self.t),
        )
        self.t = 0.0

    def run(self, gesture, point=(0.5, 0.5), seconds=DT, level=0.0):
        for _ in range(max(1, round(seconds / DT))):
            self.ctl.update(gesture, point, self.t, level)
            self.t += DT

    def tap(self, gesture, seconds=0.12):
        """Bend for ``seconds`` and return to the L pose."""
        self.run(gesture, seconds=seconds)
        self.run(L)


def test_default_gesture_set():
    assert LEFT is G.THUMB_BENT and RIGHT is G.INDEX_BENT
    assert G(config.CURSOR_GESTURE) is G.L_POSE and G(config.KEYBOARD_GESTURE) is G.SHAKA


# cursor clutch
def test_l_pose_moves_cursor():
    r = Rig()
    r.run(L, (0.2, 0.3), seconds=1.0)
    assert r.cursor.target == pytest.approx((0.2 * (W - 1), 0.3 * (H - 1)), abs=2)
    assert r.buttons.events == []


@pytest.mark.parametrize("other", [G.OPEN_PALM, G.NONE, G.SHAKA])
def test_other_poses_do_not_move_cursor(other):
    r = Rig()
    r.run(L, (0.5, 0.5), seconds=0.5)
    n = len(r.cursor.events)
    r.run(other, (0.1, 0.9), seconds=0.5)
    assert not r.cursor.moved(n)


def test_cursor_freezes_while_click_pose_builds_up():
    r = Rig()
    r.run(L, (0.5, 0.5), seconds=0.5)
    n = len(r.cursor.events)
    r.run(L, (0.55, 0.5), seconds=0.3, level=0.7)
    assert not r.cursor.moved(n)
    r.run(L, (0.6, 0.5), seconds=0.5)
    assert r.cursor.moved(n)


# left click
def test_thumb_bend_and_return_is_left_click():
    r = Rig()
    r.run(L, seconds=0.5)
    r.run(LEFT, seconds=0.12)
    assert r.buttons.events == []  # nothing pressed while bent
    r.run(L)
    assert r.buttons.events == [("click", "left")]
    assert r.ctl.last_event[0] == "LEFT CLICK"


def test_click_lands_at_pre_bend_position():
    r = Rig()
    r.run(L, (0.5, 0.5), seconds=1.0)
    r.run(L, (0.52, 0.52))  # tiny drift as the thumb starts bending
    r.tap(LEFT)
    jump = next(e for e in r.cursor.events if e[0] == "jump")
    assert jump[1] == pytest.approx((500, 500), abs=3)


def test_two_quick_bends_double_click_in_one_spot():
    r = Rig()
    r.run(L, seconds=1.0)
    r.tap(LEFT)
    r.run(L, seconds=0.1)
    r.tap(LEFT)
    assert r.buttons.events == [("click", "left"), ("click", "left")]
    assert len([e for e in r.cursor.events if e[0] == "jump"]) == 1


def test_bend_into_fist_never_clicks():
    r = Rig()
    r.run(L, seconds=0.5)
    r.run(LEFT, seconds=0.1)
    r.run(RIGHT, seconds=0.1)
    r.run(G.FIST, seconds=0.3)
    r.run(L, seconds=0.3)  # opening again soon after the fist
    assert r.buttons.events == []


def test_clicks_ignored_shortly_after_fist():
    r = Rig()
    r.run(G.FIST, seconds=0.2)
    r.run(L, seconds=0.1)
    r.tap(LEFT)
    assert r.buttons.events == []
    r.run(L, seconds=config.AFTER_FIST_CLICK_GUARD_S)
    r.tap(LEFT)
    assert r.buttons.events == [("click", "left")]


def test_flicker_shorter_than_min_hold_does_not_click(monkeypatch):
    monkeypatch.setattr(config, "CLICK_MIN_HOLD_S", 0.1)
    r = Rig()
    r.run(L, seconds=0.5)
    r.tap(LEFT, seconds=DT)
    assert r.buttons.events == []


# drag
def test_hold_arms_drag_then_moving_drags():
    r = Rig()
    r.run(L, (0.3, 0.3), seconds=0.5)
    r.run(LEFT, (0.3, 0.3), seconds=config.DRAG_ARM_S + 0.1)
    assert r.buttons.events == [("press", "left")]
    assert r.ctl.last_event[0] == "DRAG READY"
    r.run(LEFT, (0.6, 0.6), seconds=0.5)
    assert r.ctl.dragging and r.cursor.target[0] > 0.45 * W
    r.run(L, (0.6, 0.6))
    assert r.buttons.events == [("press", "left"), ("release", "left")]
    assert r.ctl.last_event[0] == "DROP"


def test_false_quick_bend_while_moving_never_drags():
    r = Rig()
    r.run(L, (0.3, 0.3), seconds=0.5)
    for i in range(int(config.DRAG_ARM_S / DT) - 2):
        r.run(LEFT, (0.3 + i * 0.03, 0.3))
    r.run(L, (0.6, 0.3))
    assert [e[0] for e in r.buttons.events] == ["click"]


def test_armed_but_still_is_auto_released():
    r = Rig()
    r.run(L, seconds=0.5)
    r.run(LEFT, seconds=config.DRAG_ARM_S + config.LEFT_HOLD_STILL_MAX_S + 0.2)
    assert r.ctl.last_event[0] == "AUTO RELEASE"
    r.run(L)
    assert r.buttons.events == [("press", "left"), ("release", "left")]


def test_dragging_is_never_auto_released():
    r = Rig()
    r.run(L, (0.3, 0.3), seconds=0.5)
    r.run(LEFT, (0.3, 0.3), seconds=config.DRAG_ARM_S + 0.1)
    r.run(LEFT, (0.6, 0.6), seconds=config.LEFT_HOLD_STILL_MAX_S + 0.5)
    assert r.ctl.dragging and r.buttons.events == [("press", "left")]


def test_lost_hand_while_dragging_releases():
    r = Rig()
    r.run(L, seconds=0.5)
    r.run(LEFT, seconds=config.DRAG_ARM_S + 0.1)
    r.run(LEFT, point=None, seconds=config.HAND_LOST_RELEASE_S / 2)
    assert r.buttons.events == [("press", "left")]  # brief dropout keeps the drag
    r.run(LEFT, point=None, seconds=config.HAND_LOST_RELEASE_S)
    assert r.buttons.events == [("press", "left"), ("release", "left")]


def test_release_all_lets_go_of_button():
    r = Rig()
    r.run(L, seconds=0.5)
    r.run(LEFT, seconds=config.DRAG_ARM_S + 0.1)
    r.ctl.release_all()
    r.ctl.release_all()
    assert r.buttons.events == [("press", "left"), ("release", "left")]


# right click
def test_index_bend_and_return_is_right_click():
    r = Rig()
    r.run(L, seconds=0.5)
    r.run(RIGHT, seconds=0.2)
    assert r.buttons.events == []
    r.run(L)
    assert r.buttons.events == [("click", "right")]
    assert r.ctl.last_event[0] == "RIGHT CLICK"


def test_right_bend_not_returning_to_l_does_not_click():
    r = Rig()
    r.run(L, seconds=0.5)
    r.run(RIGHT, seconds=0.2)
    r.run(G.OPEN_PALM, seconds=0.2)
    assert r.buttons.events == []


# scroll
def scrolled(r):
    return sum(e[1] for e in r.buttons.events if e[0] == "scroll")


def test_auto_scroll_keeps_scrolling_while_hand_is_held_above_anchor(monkeypatch):
    monkeypatch.setattr(config, "SCROLL_MODE", "auto")
    r = Rig()
    r.run(L, seconds=0.3)
    r.run(G.TWO_FINGERS, (0.5, 0.5))
    offset = config.SCROLL_AUTO_DEADZONE + 0.1
    r.run(G.TWO_FINGERS, (0.5, 0.5 - offset), seconds=1.0)
    first = scrolled(r)
    r.run(G.TWO_FINGERS, (0.5, 0.5 - offset), seconds=1.0)
    assert first > 0
    assert scrolled(r) - first == pytest.approx(0.1 * config.SCROLL_AUTO_SPEED, rel=0.1)


def test_auto_scroll_dead_zone_and_stop(monkeypatch):
    monkeypatch.setattr(config, "SCROLL_MODE", "auto")
    r = Rig()
    r.run(G.TWO_FINGERS, (0.5, 0.5))
    r.run(G.TWO_FINGERS, (0.5, 0.5 - config.SCROLL_AUTO_DEADZONE * 0.8), seconds=1.0)
    assert scrolled(r) == 0
    r.run(G.TWO_FINGERS, (0.5, 0.3), seconds=0.3)
    n = len(r.buttons.events)
    r.run(L, (0.5, 0.3), seconds=0.5)
    assert len(r.buttons.events) == n and r.ctl.scroll_anchor is None


def test_scroll_freezes_cursor():
    r = Rig()
    r.run(L, seconds=0.5)
    n = len(r.cursor.events)
    r.run(G.TWO_FINGERS, (0.5, 0.3), seconds=0.5)
    assert not r.cursor.moved(n)


# pause and keyboard
def test_fist_hold_toggles_pause():
    r = Rig()
    r.run(G.FIST, seconds=config.FIST_HOLD_S + 0.1)
    assert r.ctl.paused
    n = len(r.cursor.events)
    r.run(L, (0.1, 0.1), seconds=0.6)
    r.tap(LEFT)
    assert r.buttons.events == [] and not r.cursor.moved(n)
    r.run(G.FIST, seconds=config.FIST_HOLD_S + 0.1)
    assert not r.ctl.paused


def test_shaka_hold_toggles_keyboard_once():
    r = Rig()
    r.run(G.SHAKA, seconds=config.KEYBOARD_TOGGLE_HOLD_S * 3)
    assert len(r.keyboard_calls) == 1 and r.ctl.last_event[0] == "KEYBOARD"


def test_open_palm_does_not_open_keyboard():
    r = Rig()
    r.run(G.OPEN_PALM, seconds=config.KEYBOARD_TOGGLE_HOLD_S * 3)
    assert r.keyboard_calls == []


def test_short_shaka_does_nothing_and_second_hold_toggles_again():
    r = Rig()
    r.run(G.SHAKA, seconds=config.KEYBOARD_TOGGLE_HOLD_S / 2)
    r.run(L, seconds=0.2)
    assert r.keyboard_calls == []
    for _ in range(2):
        r.run(G.SHAKA, seconds=config.KEYBOARD_TOGGLE_HOLD_S + 0.1)
        r.run(L, seconds=0.2)
    assert len(r.keyboard_calls) == 2


def test_shaka_ignored_while_paused():
    r = Rig()
    r.run(G.FIST, seconds=config.FIST_HOLD_S + 0.1)
    r.run(G.SHAKA, seconds=config.KEYBOARD_TOGGLE_HOLD_S + 0.5)
    assert r.keyboard_calls == []


# detector + controller together
def run_detector_and_controller(frames):
    r = Rig()
    d = GestureDetector()
    for lm in frames:
        g = d.update(lm)
        r.ctl.update(g, (0.5, 0.5), r.t, max(d.strength.values()))
        r.t += DT
    return r


def test_thumb_bend_tap_is_left_click_end_to_end():
    r = run_detector_and_controller(
        [pose("l_pose")] * 20 + [pose("thumb_bent")] * 4 + [pose("l_pose")] * 10
    )
    assert r.buttons.events == [("click", "left")]


def test_index_bend_tap_is_right_click_end_to_end():
    r = run_detector_and_controller(
        [pose("l_pose")] * 20 + [pose("index_bent")] * 4 + [pose("l_pose")] * 10
    )
    assert r.buttons.events == [("click", "right")]


def test_closing_into_fist_end_to_end_never_clicks():
    r = run_detector_and_controller(
        [pose("l_pose")] * 20 + [pose("thumb_bent")] * 3 + [pose("fist")] * 30
        + [pose("index_bent")] * 3 + [pose("l_pose")] * 10
    )
    assert r.buttons.events == []
