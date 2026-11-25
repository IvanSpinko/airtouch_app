import json

import pytest

from gesture_control import config, settings


@pytest.fixture(autouse=True)
def _restore_config():
    before = settings.current()
    yield
    settings.apply(before)


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "s.json"
    settings.save({"ACCEL_MAX_GAIN": 4.2, "CAMERA_INDEX": 1, "SHOW_PREVIEW": False}, path)
    settings.apply(settings.DEFAULTS)
    assert config.ACCEL_MAX_GAIN == settings.DEFAULTS["ACCEL_MAX_GAIN"]
    settings.load(path)
    assert config.ACCEL_MAX_GAIN == 4.2 and config.CAMERA_INDEX == 1 and config.SHOW_PREVIEW is False


def test_values_are_clamped_and_typed():
    applied = settings.apply({"ACCEL_MAX_GAIN": 100, "CAMERA_INDEX": 2.7, "FIST_HOLD_S": -1})
    assert applied == {"ACCEL_MAX_GAIN": 6.0, "CAMERA_INDEX": 3, "FIST_HOLD_S": 0.4}
    assert isinstance(config.CAMERA_INDEX, int)


def test_unknown_and_invalid_values_are_ignored():
    applied = settings.apply({"NOT_A_SETTING": 1, "ACCEL_MIN_GAIN": "fast", "HOTKEY_TOGGLE": "x"})
    assert applied == {}
    assert config.HOTKEY_TOGGLE == "<ctrl>+<alt>+g"


def test_missing_or_broken_file_changes_nothing(tmp_path):
    assert settings.load(tmp_path / "missing.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert settings.load(broken) == {}
    broken.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert settings.load(broken) == {}


def test_reset_restores_defaults(tmp_path):
    path = tmp_path / "s.json"
    settings.save({"ACCEL_MAX_GAIN": 5.5}, path)
    settings.reset(path)
    assert json.loads(path.read_text())["ACCEL_MAX_GAIN"] == settings.DEFAULTS["ACCEL_MAX_GAIN"]


def test_every_setting_exists_in_config_within_range():
    for s in settings.SETTINGS:
        value = settings.DEFAULTS[s.key]
        if s.kind is not bool:
            assert s.minimum <= value <= s.maximum, s.key
