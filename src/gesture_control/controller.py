"""Controller: decides what to do with the tracked hand.

Stage 1 only has two states — control enabled or disabled. Mouse/keyboard/pause
modes driven by gestures are added in later stages.

The enabled flag is flipped from the hotkey thread and read from the main loop,
so it is guarded by a lock.

Check: ``python -m gesture_control.controller`` prints state changes on Ctrl+Alt+G.
"""

from __future__ import annotations

import threading
from typing import Callable

from gesture_control import config

Listener = Callable[[bool], None]


class ControlState:
    """Thread-safe on/off switch for cursor control."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()
        self._listeners: list[Listener] = []

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def set_enabled(self, value: bool) -> None:
        with self._lock:
            changed = self._enabled != value
            self._enabled = value
        if changed:
            for listener in self._listeners:
                listener(value)

    def toggle(self) -> bool:
        with self._lock:
            self._enabled = not self._enabled
            value = self._enabled
        for listener in self._listeners:
            listener(value)
        return value


class HotkeyListener:
    """Global hotkey (works while other windows are focused) that toggles control."""

    def __init__(self, state: ControlState, combo: str = config.HOTKEY_TOGGLE) -> None:
        from pynput.keyboard import GlobalHotKeys

        self._hotkeys = GlobalHotKeys({combo: state.toggle})

    def start(self) -> "HotkeyListener":
        self._hotkeys.start()
        return self

    def stop(self) -> None:
        self._hotkeys.stop()

    def __enter__(self) -> "HotkeyListener":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _demo() -> None:
    state = ControlState()
    state.subscribe(lambda on: print("control", "ENABLED" if on else "DISABLED"))
    print(f"Press {config.HOTKEY_TOGGLE} to toggle, Ctrl+C to exit.")
    with HotkeyListener(state) as hk:
        try:
            hk._hotkeys.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    _demo()
