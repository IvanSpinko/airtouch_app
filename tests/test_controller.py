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
