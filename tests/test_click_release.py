"""Default CLICK_MODE "release": quick pinch = click on opening, hold = drag."""

import pytest
from handgen import make_hand
from test_controller import DT, LEFT, RIGHT, Rig

from gesture_control import config
from gesture_control.gestures import Gesture as G
from gesture_control.gestures import GestureDetector


@pytest.fixture(autouse=True)
def _release_mode(monkeypatch):
    monkeypatch.setattr(config, "CLICK_MODE", "release")


def test_quick_pinch_clicks_on_release_without_holding_button():
    r = Rig()
    r.run(G.POINT, seconds=0.5)
    r.run(LEFT, seconds=0.15)
    assert r.buttons.events == []  # nothing pressed while pinched
    r.run(G.POINT)
    assert r.buttons.events == [("click", "left")]
    assert r.ctl.last_event[0] == "LEFT CLICK"


def test_false_quick_pinch_while_moving_never_drags_or_selects():
    r = Rig()
    r.run(G.POINT, (0.3, 0.3), seconds=0.5)
    for i in range(int(config.DRAG_ARM_S / DT) - 2):
        r.run(LEFT, (0.3 + i * 0.03, 0.3))  # hand keeps moving during a false pinch
    r.run(G.POINT, (0.6, 0.3))
    assert [e[0] for e in r.buttons.events] == ["click"]  # never a press/drag


def test_hold_arms_drag_then_moving_drags():
    r = Rig()
    r.run(G.POINT, (0.3, 0.3), seconds=0.5)
    r.run(LEFT, (0.3, 0.3), seconds=config.DRAG_ARM_S + 0.1)
    assert r.buttons.events == [("press", "left")]
    assert r.ctl.last_event[0] == "DRAG READY"
    r.run(LEFT, (0.6, 0.6), seconds=0.5)
    assert r.ctl.dragging
    r.run(G.POINT, (0.6, 0.6))
    assert r.buttons.events == [("press", "left"), ("release", "left")]
    assert r.ctl.last_event[0] == "DROP"


def test_two_quick_taps_double_click_in_one_spot():
    r = Rig()
    r.run(G.POINT, (0.5, 0.5), seconds=1.0)
    for _ in range(2):
        r.run(LEFT, seconds=0.12)
        r.run(G.POINT, seconds=0.12)
    assert r.buttons.events == [("click", "left"), ("click", "left")]
    assert len([e for e in r.cursor.events if e[0] == "jump"]) == 1


def test_armed_but_still_is_auto_released_and_no_extra_click():
    r = Rig()
    r.run(G.POINT, seconds=0.5)
    r.run(LEFT, seconds=config.DRAG_ARM_S + config.LEFT_HOLD_STILL_MAX_S + 0.2)
    r.run(G.POINT)
    assert r.buttons.events == [("press", "left"), ("release", "left")]


def test_lost_hand_while_armed_releases():
    r = Rig()
    r.run(G.POINT, seconds=0.5)
    r.run(LEFT, seconds=config.DRAG_ARM_S + 0.1)
    r.run(LEFT, point=None, seconds=config.HAND_LOST_RELEASE_S + 0.1)
    assert r.buttons.events == [("press", "left"), ("release", "left")]


def test_cursor_freezes_while_pinch_builds_up():
    r = Rig()
    r.run(G.POINT, (0.5, 0.5), seconds=0.5)
    n = len(r.cursor.events)
    for i in range(10):  # strength rising, not yet a pinch; hand drifts
        r.ctl.update(G.POINT, (0.5 + i * 0.01, 0.5), r.t, pinch_level=0.7)
        r.t += DT
    assert not any(e[0] == "target" for e in r.cursor.events[n:])
    r.run(G.POINT, (0.6, 0.5), seconds=0.5)  # pinch strength back to 0
    assert any(e[0] == "target" for e in r.cursor.events[n:])


def test_right_click_unchanged_in_release_mode():
    r = Rig()
    r.run(G.POINT, seconds=0.5)
    r.run(RIGHT, seconds=0.3)
    r.run(G.POINT)
    assert r.buttons.events == [("click", "right")]


def test_quick_tap_end_to_end_with_detector():
    r = Rig()
    d = GestureDetector()
    point = make_hand(("index",))
    pinch = make_hand(("index",), thumb_tip="index")
    for lm in [point] * 15 + [pinch] * 3 + [point] * 10:
        g = d.update(lm)
        r.ctl.update(g, (0.5, 0.5), r.t, max(d.strength.values()))
        r.t += DT
    assert r.buttons.events == [("click", "left")]
