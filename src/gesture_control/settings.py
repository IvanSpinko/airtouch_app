"""User settings: the few values worth changing without editing config.py.

They are stored in ``%APPDATA%\\AirTouch\\settings.json`` and applied on top of the
defaults in config.py at start-up (and again whenever the settings window or the
calibration saves them). Unknown keys and out-of-range values are ignored /
clamped, so a damaged file cannot break the app.

Check: ``python -m gesture_control.settings`` prints the current values and file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gesture_control import config


@dataclass(frozen=True)
class Setting:
    key: str          # name of the value in config.py
    label: str        # shown in the settings window
    minimum: float
    maximum: float
    step: float
    kind: type = float

    def clamp(self, value):
        if self.kind is bool:
            return bool(value)
        v = min(max(float(value), self.minimum), self.maximum)
        return int(round(v)) if self.kind is int else v


SETTINGS: tuple[Setting, ...] = (
    Setting("ACCEL_MAX_GAIN", "Cursor speed (fast moves)", 1.0, 6.0, 0.1),
    Setting("ACCEL_MIN_GAIN", "Aiming speed (slow moves)", 0.1, 1.0, 0.05),
    Setting("CURSOR_SENSITIVITY", "Base sensitivity (calibration)", 0.5, 4.0, 0.05),
    Setting("CURSOR_GLIDE_TAU_S", "Smoothness (glide time, s)", 0.01, 0.15, 0.005),
    Setting("FILTER_MIN_CUTOFF", "Jitter filter (lower = steadier)", 0.1, 1.5, 0.05),
    Setting("CURSOR_DEADZONE_PX", "Dead zone (px)", 0.0, 10.0, 0.5),
    Setting("DRAG_ARM_S", "Hold thumb to start drag (s)", 0.2, 1.5, 0.05),
    Setting("KEYBOARD_TOGGLE_HOLD_S", "Hold shaka for keyboard (s)", 0.5, 3.0, 0.1),
    Setting("FIST_HOLD_S", "Hold fist to pause (s)", 0.4, 3.0, 0.1),
    Setting("SCROLL_AUTO_SPEED", "Scroll speed", 20.0, 200.0, 5.0),
    Setting("CAMERA_INDEX", "Camera number", 0, 5, 1, int),
    Setting("SHOW_PREVIEW", "Show camera window", 0, 1, 1, bool),
    Setting("CUSTOM_KEYBOARD", "AirTouch keyboard (off = Windows osk.exe)", 0, 1, 1, bool),
)
BY_KEY = {s.key: s for s in SETTINGS}
# config.py values as shipped, for "Reset to defaults".
DEFAULTS = {s.key: getattr(config, s.key) for s in SETTINGS}


def current() -> dict:
    return {s.key: getattr(config, s.key) for s in SETTINGS}


def apply(values: dict) -> dict:
    """Validates ``values`` and writes them into config; returns what was applied."""
    applied = {}
    for key, value in values.items():
        setting = BY_KEY.get(key)
        if setting is None:
            continue
        try:
            applied[key] = setting.clamp(value)
        except (TypeError, ValueError):
            continue
        setattr(config, key, applied[key])
    return applied


def load(path: Path | None = None) -> dict:
    """Reads the settings file and applies it; a missing or broken file changes nothing."""
    path = config.SETTINGS_PATH if path is None else path
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return apply(data) if isinstance(data, dict) else {}


def save(values: dict | None = None, path: Path | None = None) -> Path:
    """Applies ``values`` (default: the current ones) and writes them to the file."""
    path = config.SETTINGS_PATH if path is None else path
    applied = apply(values) if values is not None else {}
    data = {**current(), **applied}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def reset(path: Path | None = None) -> Path:
    return save(dict(DEFAULTS), path)


if __name__ == "__main__":
    load()
    print(f"settings file: {config.SETTINGS_PATH}")
    for s in SETTINGS:
        print(f"  {s.label:34s} {getattr(config, s.key)}")
