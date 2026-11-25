"""Settings and calibration windows (PyQt6), opened from the tray menu."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from gesture_control import config, settings
from gesture_control.calibration import CalibrationResult, calibrate


class SettingsDialog(QDialog):
    """Form with every user setting; Save writes settings.json and calls ``on_saved``."""

    def __init__(self, on_saved: Callable[[], None] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.on_saved = on_saved
        self.setWindowTitle(f"{config.APP_NAME} settings")
        self.fields: dict[str, QWidget] = {}
        form = QFormLayout()
        for s in settings.SETTINGS:
            form.addRow(s.label, self._field(s))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            lambda: self.set_values(settings.DEFAULTS))
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel(f"Saved to {config.SETTINGS_PATH}"))
        layout.addWidget(buttons)
        self.set_values(settings.current())

    def _field(self, s: settings.Setting) -> QWidget:
        if s.kind is bool:
            w = QCheckBox()
        elif s.kind is int:
            w = QSpinBox()
            w.setRange(int(s.minimum), int(s.maximum))
        else:
            w = QDoubleSpinBox()
            w.setRange(s.minimum, s.maximum)
            w.setSingleStep(s.step)
            w.setDecimals(config.SETTINGS_DECIMALS)
        self.fields[s.key] = w
        return w

    def set_values(self, values: dict) -> None:
        for key, w in self.fields.items():
            if key not in values:
                continue
            if isinstance(w, QCheckBox):
                w.setChecked(bool(values[key]))
            else:
                w.setValue(values[key])

    def values(self) -> dict:
        return {
            key: (w.isChecked() if isinstance(w, QCheckBox) else w.value())
            for key, w in self.fields.items()
        }

    def save(self) -> None:
        settings.save(self.values())
        if self.on_saved is not None:
            self.on_saved()
        self.accept()


class CalibrationDialog(QDialog):
    """Two-step calibration fed by the engine snapshot (cursor control is off meanwhile)."""

    STEPS = (
        ("Step 1 of 2: show the L pose and slowly sweep it over the whole area where "
         "moving your hand is comfortable - left, right, up and down.", config.CALIB_MOVE_S),
        ("Step 2 of 2: hold the L pose still in the middle of that area.", config.CALIB_HOLD_S),
    )

    def __init__(self, engine, screen: tuple[int, int],
                 on_saved: Callable[[], None] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.screen = screen
        self.on_saved = on_saved
        self.result: CalibrationResult | None = None
        self.setWindowTitle(f"{config.APP_NAME} calibration")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.text = QLabel("Cursor control is paused during calibration. Press Start.")
        self.text.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, config.CALIB_PROGRESS_STEPS)
        self.start_btn = QPushButton("Start")
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        cancel = QPushButton("Close")
        self.start_btn.clicked.connect(self.start)
        self.save_btn.clicked.connect(self.save)
        cancel.clicked.connect(self.reject)
        row = QHBoxLayout()
        for b in (self.start_btn, self.save_btn, cancel):
            row.addWidget(b)
        layout = QVBoxLayout(self)
        layout.addWidget(self.text)
        layout.addWidget(self.progress)
        layout.addLayout(row)
        self.resize(config.CALIB_DIALOG_WIDTH, self.sizeHint().height())

        self._was_enabled = engine.state.enabled
        engine.state.set_enabled(False)
        self._samples: list[list[tuple[float, float]]] = [[], []]
        self._step = -1
        self._elapsed = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._samples = [[], []]
        self._step, self._elapsed = 0, 0.0
        self.result = None
        self.save_btn.setEnabled(False)
        self.start_btn.setText("Restart")
        self.text.setText(self.STEPS[0][0])
        self.timer.start(config.CALIB_SAMPLE_MS)

    def _tick(self) -> None:
        point = self.engine.snapshot().cursor_point
        if point is not None:
            self._samples[self._step].append(point)
            self._elapsed += config.CALIB_SAMPLE_MS / 1000.0
        duration = self.STEPS[self._step][1]
        self.progress.setValue(int(min(self._elapsed / duration, 1.0) * config.CALIB_PROGRESS_STEPS))
        if self._elapsed < duration:
            return
        if self._step == 0:
            self._step, self._elapsed = 1, 0.0
            self.text.setText(self.STEPS[1][0])
            return
        self.timer.stop()
        self.finish()

    def finish(self) -> None:
        try:
            self.result = calibrate(np.array(self._samples[0]), np.array(self._samples[1]),
                                    *self.screen)
        except ValueError as exc:
            self.text.setText(f"Calibration failed: {exc}")
            return
        r = self.result
        self.text.setText(
            f"Comfortable area: {r.range_x:.0%} x {r.range_y:.0%} of the camera view, "
            f"hand tremor {r.jitter_px:.1f} px.\n"
            f"Sensitivity {r.sensitivity:.2f}, jitter filter {r.filter_min_cutoff:.2f}, "
            f"dead zone {r.deadzone_px:.1f} px. Save to apply."
        )
        self.save_btn.setEnabled(True)

    def save(self) -> None:
        if self.result is not None:
            settings.save(self.result.as_settings())
            if self.on_saved is not None:
                self.on_saved()
        self.accept()

    def done(self, code: int) -> None:
        self.timer.stop()
        self.engine.state.set_enabled(self._was_enabled)
        super().done(code)
