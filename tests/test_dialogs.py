"""Settings and calibration windows, driven without a camera (off-screen Qt)."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from gesture_control import config, settings  # noqa: E402
from gesture_control.controller import ControlState  # noqa: E402
from gesture_control.dialogs import CalibrationDialog, SettingsDialog  # noqa: E402
from gesture_control.engine import Snapshot  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    before = settings.current()
    yield
    settings.apply(before)


def test_settings_dialog_shows_and_saves_values(qapp):
    saved = []
    dlg = SettingsDialog(on_saved=lambda: saved.append(True))
    assert dlg.values()["ACCEL_MAX_GAIN"] == pytest.approx(config.ACCEL_MAX_GAIN)
    dlg.fields["ACCEL_MAX_GAIN"].setValue(4.5)
    dlg.fields["SHOW_PREVIEW"].setChecked(False)
    dlg.save()
    assert saved == [True]
    assert config.ACCEL_MAX_GAIN == 4.5 and config.SHOW_PREVIEW is False
    assert json.loads(config.SETTINGS_PATH.read_text())["ACCEL_MAX_GAIN"] == 4.5


def test_settings_dialog_restore_defaults(qapp):
    settings.apply({"ACCEL_MAX_GAIN": 5.0})
    dlg = SettingsDialog()
    dlg.set_values(settings.DEFAULTS)
    assert dlg.values()["ACCEL_MAX_GAIN"] == pytest.approx(settings.DEFAULTS["ACCEL_MAX_GAIN"])


class FakeEngine:
    """Feeds the dialog a sweep during step 1 and a still hand during step 2."""

    def __init__(self):
        self.state = ControlState()
        self.dialog = None
        self.rng = np.random.default_rng(0)

    def snapshot(self):
        if self.dialog is not None and self.dialog._step == 1:
            point = (0.5 + self.rng.normal(0, 0.001), 0.4 + self.rng.normal(0, 0.001))
        else:
            point = (0.5 + self.rng.uniform(-0.15, 0.15), 0.4 + self.rng.uniform(-0.1, 0.1))
        return Snapshot(running=True, hand_visible=True, cursor_point=point)


def run_calibration(dlg, max_ticks=1000):
    dlg.start()
    dlg.timer.stop()  # drive the ticks by hand instead of the Qt timer
    for _ in range(max_ticks):
        dlg._tick()
        if not dlg.timer.isActive() and dlg._step == 1 and (dlg.result or "failed" in dlg.text.text()):
            break


def test_calibration_runs_both_steps_and_saves(qapp, monkeypatch):
    engine = FakeEngine()
    engine.state.set_enabled(True)
    saved = []
    dlg = CalibrationDialog(engine, (2560, 1440), on_saved=lambda: saved.append(True))
    engine.dialog = dlg
    dlg.STEPS = (("move", 1.5), ("hold", 1.5))  # >= CALIB_MIN_SAMPLES samples each
    assert engine.state.enabled is False  # control is off during calibration
    run_calibration(dlg)
    assert dlg.result is not None and dlg.save_btn.isEnabled()
    assert dlg.result.range_x == pytest.approx(0.3 * 0.9, rel=0.15)
    assert dlg.result.jitter_px < 10  # the still step really was still
    dlg.save()
    assert saved == [True]
    assert config.CURSOR_SENSITIVITY == pytest.approx(dlg.result.sensitivity)
    assert engine.state.enabled is True  # restored when the window closes


def test_closing_calibration_restores_disabled_state(qapp):
    engine = FakeEngine()
    engine.state.set_enabled(False)
    dlg = CalibrationDialog(engine, (1920, 1080))
    dlg.reject()
    assert engine.state.enabled is False


def test_calibration_without_hand_reports_failure(qapp):
    class NoHand(FakeEngine):
        def snapshot(self):
            return Snapshot(running=True, hand_visible=False, cursor_point=None)

    dlg = CalibrationDialog(NoHand(), (1920, 1080))
    dlg.finish()  # no samples collected
    assert dlg.result is None and "failed" in dlg.text.text()
