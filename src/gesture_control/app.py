"""AirTouch tray app: system tray icon, status indicator and the engine in the background.

Run: ``python -m gesture_control.app`` (this is also what AirTouch.exe starts).

Tray menu: enable / disable control (same as Ctrl+Alt+G), pause, on-screen
keyboard, camera window, indicator, calibration, settings, quit. The icon color
shows the state: green = on, yellow = paused, grey = off, red = error.
"""

from __future__ import annotations

import argparse
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from gesture_control import __version__, config, keyboard, settings
from gesture_control.engine import Engine
from gesture_control.overlay import Indicator, make_icon, status_of


class TrayApp:
    def __init__(self, app: QApplication, log: bool = False) -> None:
        self.app = app
        self.log = log
        self.engine = Engine(log=log).start()
        self.indicator = Indicator()
        if config.SHOW_INDICATOR:
            self.indicator.show()
        self._last_error: str | None = None
        self._icons = {
            color: make_icon(color) for color in (
                config.STATUS_COLOR_ON, config.STATUS_COLOR_PAUSED,
                config.STATUS_COLOR_OFF, config.STATUS_COLOR_ERROR,
            )
        }
        self.tray = QSystemTrayIcon(self._icons[config.STATUS_COLOR_OFF])
        self.tray.setToolTip(config.APP_NAME)
        self.tray.setContextMenu(self._build_menu())
        self.tray.activated.connect(self._on_activated)
        self.tray.show()

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(config.UI_REFRESH_MS)

    # -- menu ------------------------------------------------------------------------
    def _action(self, menu: QMenu, text: str, slot, checkable: bool = False) -> QAction:
        action = QAction(text, menu)
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_menu(self) -> QMenu:
        menu = QMenu()
        title = QAction(f"{config.APP_NAME} {__version__}", menu)
        title.setEnabled(False)
        menu.addAction(title)
        menu.addSeparator()
        self.act_enabled = self._action(
            menu, f"Control enabled  ({config.HOTKEY_TOGGLE})", self.toggle_enabled, True)
        self.act_paused = self._action(menu, "Paused  (hold fist)", self.toggle_pause, True)
        self._action(menu, "On-screen keyboard  (hold shaka)", self.engine_keyboard)
        menu.addSeparator()
        self.act_preview = self._action(menu, "Show camera window", self.toggle_preview, True)
        self.act_indicator = self._action(menu, "Show indicator", self.toggle_indicator, True)
        menu.addSeparator()
        self._action(menu, "Calibrate...", self.open_calibration)
        self._action(menu, "Settings...", self.open_settings)
        menu.addSeparator()
        self._action(menu, "Quit", self.quit)
        self.menu = menu
        return menu

    def _on_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:  # left click
            self.toggle_enabled()

    # -- actions ---------------------------------------------------------------------
    def toggle_enabled(self) -> None:
        self.engine.toggle_enabled()
        self.refresh()

    def toggle_pause(self) -> None:
        self.engine.toggle_pause()

    def engine_keyboard(self) -> None:
        self.engine.toggle_keyboard()

    def toggle_preview(self) -> None:
        settings.save({"SHOW_PREVIEW": not config.SHOW_PREVIEW})

    def toggle_indicator(self) -> None:
        config.SHOW_INDICATOR = not config.SHOW_INDICATOR
        self.indicator.setVisible(config.SHOW_INDICATOR)

    def open_settings(self) -> None:
        from gesture_control.dialogs import SettingsDialog

        self._dialog = SettingsDialog(on_saved=self.restart_engine)
        self._dialog.show()

    def open_calibration(self) -> None:
        from gesture_control.dialogs import CalibrationDialog
        from gesture_control.mouse import enable_dpi_awareness, screen_size

        enable_dpi_awareness()
        self._dialog = CalibrationDialog(self.engine, screen_size(), on_saved=self.restart_engine)
        self._dialog.show()

    def restart_engine(self) -> None:
        """Applies settings that are read when the engine starts (camera, filters...)."""
        enabled = self.engine.state.enabled
        self.engine.stop(timeout=config.ENGINE_STOP_TIMEOUT_S)
        self.engine = Engine(log=self.log)
        self.engine.state.set_enabled(enabled)
        self.engine.start()

    def quit(self) -> None:
        self.timer.stop()
        self.engine.stop(timeout=config.ENGINE_STOP_TIMEOUT_S)
        self.tray.hide()
        self.app.quit()

    # -- periodic refresh ------------------------------------------------------------------
    def refresh(self) -> None:
        snap = self.engine.snapshot()
        text, color = status_of(snap)
        self.tray.setIcon(self._icons.get(color, self._icons[config.STATUS_COLOR_OFF]))
        hand = "hand visible" if snap.hand_visible else "no hand"
        self.tray.setToolTip(f"{config.APP_NAME} {__version__}: {text}, {hand}")
        self.act_enabled.setChecked(snap.enabled)
        self.act_paused.setChecked(snap.paused)
        self.act_preview.setChecked(config.SHOW_PREVIEW)
        self.act_indicator.setChecked(config.SHOW_INDICATOR)
        if self.indicator.isVisible():
            self.indicator.show_snapshot(snap)
        if snap.error and snap.error != self._last_error:
            self._last_error = snap.error
            self.tray.showMessage(config.APP_NAME, snap.error,
                                  QSystemTrayIcon.MessageIcon.Critical)


def self_test() -> int:
    """Loads everything the app needs without touching camera or mouse (used on the .exe)."""
    from gesture_control.gestures import GestureClassifier
    from gesture_control.tracker import HandTracker

    problems = []
    try:
        HandTracker().close()
    except Exception as exc:
        problems.append(f"hand tracker: {exc}")
    if GestureClassifier.load_if_available() is None:
        problems.append("gesture model missing or for another gesture set")
    app = QApplication.instance() or QApplication(sys.argv)
    if not QSystemTrayIcon.isSystemTrayAvailable():
        problems.append("system tray not available")
    make_icon(config.STATUS_COLOR_ON)
    report = "OK" if not problems else "FAIL: " + "; ".join(problems)
    config.APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.APP_DATA_DIR / config.SELF_TEST_FILE).write_text(report, encoding="utf-8")
    print(report)
    del app
    return 0 if not problems else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="AirTouch tray app")
    parser.add_argument("--version", action="version", version=f"{config.APP_NAME} {__version__}")
    parser.add_argument("--log", action="store_true", help="record a session log in data/")
    parser.add_argument("--no-admin", action="store_true",
                        help="do not ask for administrator rights (keyboard clicks will not work)")
    parser.add_argument("--self-test", action="store_true",
                        help="check that models and Qt load, write the result and exit")
    args = parser.parse_args()
    settings.load()
    if args.self_test:
        sys.exit(self_test())
    if config.RUN_AS_ADMIN and not args.no_admin and not keyboard.is_admin():
        # osk.exe ignores input from normal-rights processes (UIPI), see keyboard.py.
        if keyboard.relaunch_as_admin(sys.argv[1:], module="gesture_control.app"):
            return
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(make_icon(config.STATUS_COLOR_ON))
    tray_app = TrayApp(app, log=args.log)  # noqa: F841 - keeps the tray alive
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
