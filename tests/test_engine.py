"""Engine lifecycle without a camera or mouse."""

import time

from gesture_control import engine as engine_mod
from gesture_control.camera import CameraError
from gesture_control.engine import Engine, Snapshot


def test_initial_snapshot():
    s = Engine().snapshot()
    assert isinstance(s, Snapshot)
    assert not s.running and s.enabled and s.error is None


def test_camera_error_is_reported_not_swallowed(monkeypatch):
    def broken_camera(*a, **k):
        raise CameraError("Cannot open camera #0")

    monkeypatch.setattr(engine_mod, "Camera", broken_camera)
    monkeypatch.setattr(engine_mod, "screen_size", lambda: (1920, 1080))
    monkeypatch.setattr(engine_mod, "enable_dpi_awareness", lambda: None)
    monkeypatch.setattr(engine_mod, "GestureClassifier", type(
        "NoModel", (), {"load_if_available": staticmethod(lambda: None)}))

    class FakeMouse:
        position = (0, 0)

    monkeypatch.setattr(engine_mod, "Mouse", FakeMouse)
    eng = Engine().start()
    deadline = time.monotonic() + 5
    while eng.snapshot().error is None and time.monotonic() < deadline:
        time.sleep(0.02)
    eng.stop(timeout=2)
    s = eng.snapshot()
    assert not s.running
    assert "Cannot open camera" in s.error


def test_enable_toggle_is_immediate():
    eng = Engine()
    assert eng.toggle_enabled() is False
    assert eng.state.enabled is False
