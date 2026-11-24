"""Controller: decides what to do with the tracked hand.

- ``ControlState`` — global on/off switch flipped by the Ctrl+Alt+G hotkey (from the
  hotkey thread, so it is guarded by a lock).
- ``GestureController`` — state machine that turns gestures into clicks, drags and
  scrolling.

Check: ``python -m gesture_control.controller`` prints state changes on Ctrl+Alt+G.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Callable, Protocol

from gesture_control import config
from gesture_control.gestures import Gesture
from gesture_control.mouse import CursorMapper

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


class CursorSink(Protocol):
    def set_target(self, x: float, y: float) -> None: ...
    def jump_to(self, x: float, y: float) -> None: ...
    def stop_moving(self) -> None: ...


class ButtonSink(Protocol):
    def press(self, button: str) -> None: ...
    def release(self, button: str) -> None: ...
    def click(self, button: str) -> None: ...
    def scroll(self, ticks: float) -> None: ...


class GestureController:
    """Turns recognized gestures into mouse actions.

    - POINT / NONE / OPEN_PALM: the cursor follows the index fingertip.
    Which pinch is the left / right button is set in config (``LEFT_BUTTON_GESTURE``,
    ``RIGHT_BUTTON_GESTURE``); by default thumb + middle is left, so the index finger
    that steers the cursor stays still while clicking.

    - left pinch: button down at the pre-pinch position; moving the hand past
      ``DRAG_START_PX`` drags; opening the pinch releases. Two quick pinches land in
      the same spot, which Windows turns into a double click. Dragging needs the
      pinch held for ``DRAG_MIN_HOLD_S``, so a quick pinch never selects anything.
    - right pinch: right click; the cursor stays put while pinched.
    - TWO_FINGERS: cursor frozen, vertical hand movement scrolls.
    - FIST held for ``FIST_HOLD_S``: toggles pause.

    OS output goes through ``cursor`` and ``buttons``, so the logic is testable
    without touching the real mouse.
    """

    def __init__(self, mapper: CursorMapper, cursor: CursorSink, buttons: ButtonSink) -> None:
        self.mapper = mapper
        self.cursor = cursor
        self.buttons = buttons
        self.paused = False
        self.left_gesture = Gesture(config.LEFT_BUTTON_GESTURE)
        self.right_gesture = Gesture(config.RIGHT_BUTTON_GESTURE)
        self._gesture = Gesture.NONE
        self._history: deque[tuple[float, tuple[int, int]]] = deque()
        self._target: tuple[int, int] | None = None
        self._last_seen: float | None = None
        self._left_down = False
        self._dragging = False
        self._pinch_anchor: tuple[float, float] = (0.0, 0.0)
        self._pinch_since = 0.0
        self._scrolling = False
        self._settle_until = 0.0
        self._scroll_ref_y = 0.0
        self._fist_since: float | None = None
        self._fist_used = False

    @property
    def gesture(self) -> Gesture:
        return self._gesture

    @property
    def dragging(self) -> bool:
        return self._dragging

    # -- public API -------------------------------------------------------------
    def release_all(self) -> None:
        """Releases held buttons and stops cursor motion (on disable, quit, lost hand)."""
        if self._left_down:
            self.buttons.release("left")
        self._left_down = False
        self._dragging = False
        self._gesture = Gesture.NONE
        self.cursor.stop_moving()

    def update(self, gesture: Gesture, point: tuple[float, float] | None, t: float) -> None:
        """Processes one frame. ``point`` is the normalized cursor landmark or None."""
        if point is None:
            if self._last_seen is not None and t - self._last_seen > config.HAND_LOST_RELEASE_S:
                if self._left_down or self._gesture is not Gesture.NONE:
                    self.release_all()
            return
        self._last_seen = t

        self._update_pause(gesture, t)
        if self.paused or gesture is Gesture.FIST:
            if self._left_down or self._gesture not in (Gesture.NONE, Gesture.FIST):
                self.release_all()
            self._gesture = gesture
            return

        x, y = point
        raw = self.mapper.zone.to_screen(x, y, self.mapper.width, self.mapper.height)
        new_target = self.mapper.update(x, y, t)
        if new_target is not None:
            self._target = new_target
            self._history.append((t, new_target))
            while self._history and t - self._history[0][0] > config.CURSOR_HISTORY_S:
                self._history.popleft()
        if self._target is None:
            self._target = (round(raw[0]), round(raw[1]))

        prev, self._gesture = self._gesture, gesture
        if gesture is not prev:
            self._exit(prev, t)
            self._enter(gesture, y, t)
        self._hold(gesture, y, t)

    # -- state machine ----------------------------------------------------------
    def _update_pause(self, gesture: Gesture, t: float) -> None:
        if gesture is not Gesture.FIST:
            self._fist_since = None
            self._fist_used = False
            return
        if self._fist_since is None:
            self._fist_since = t
        if not self._fist_used and t - self._fist_since >= config.FIST_HOLD_S:
            self.paused = not self.paused
            self._fist_used = True

    def _anchor(self, t: float) -> tuple[int, int]:
        """Cursor position from CLICK_LOOKBACK_S ago (before the fingers started closing)."""
        anchor = self._history[0][1] if self._history else self._target
        for ts, pos in self._history:
            if ts <= t - config.CLICK_LOOKBACK_S:
                anchor = pos
            else:
                break
        return anchor

    def _enter(self, gesture: Gesture, y: float, t: float) -> None:
        if gesture in (self.left_gesture, self.right_gesture):
            # Repeated pinch during the settle window reuses the same spot (double click).
            if t >= self._settle_until:
                ax, ay = self._anchor(t)
                self.cursor.jump_to(ax, ay)
            self._pinch_anchor = self._target
            self._pinch_since = t
            if gesture is self.left_gesture:
                self.buttons.press("left")
                self._left_down = True
                self._dragging = False
            else:
                self.buttons.click("right")
        elif gesture is Gesture.TWO_FINGERS:
            self.cursor.stop_moving()
            self._scroll_ref_y = y
            self._scrolling = False

    def _exit(self, gesture: Gesture, t: float) -> None:
        if gesture is self.left_gesture and self._left_down:
            self.buttons.release("left")
            self._left_down = False
            if not self._dragging:
                self._settle_until = t + config.CLICK_SETTLE_S
            self._dragging = False
        elif gesture is self.right_gesture:
            self._settle_until = t + config.CLICK_SETTLE_S

    def _moved_from_anchor(self) -> bool:
        # Smoothed position: raw landmarks jitter by several screen pixels per frame.
        dx = self._target[0] - self._pinch_anchor[0]
        dy = self._target[1] - self._pinch_anchor[1]
        return math.hypot(dx, dy) > config.DRAG_START_PX

    def _hold(self, gesture: Gesture, y: float, t: float) -> None:
        if gesture is self.left_gesture:
            held = t - self._pinch_since >= config.DRAG_MIN_HOLD_S
            if not self._dragging and held and self._moved_from_anchor():
                self._dragging = True
            if self._dragging:
                self.cursor.set_target(*self._target)
        elif gesture is self.right_gesture:
            pass  # frozen
        elif gesture is Gesture.TWO_FINGERS:
            dy = self._scroll_ref_y - y  # hand up -> positive
            if not self._scrolling:
                if abs(dy) < config.SCROLL_START_RATIO:
                    return
                self._scrolling = True
                self._scroll_ref_y = y  # the dead-band itself does not scroll
                return
            ticks = dy * config.SCROLL_TICKS_PER_FRAME
            if config.SCROLL_INVERT:
                ticks = -ticks
            if abs(ticks) >= config.SCROLL_MIN_TICKS:
                self.buttons.scroll(ticks)
                self._scroll_ref_y = y
        else:
            if t < self._settle_until and not self._moved_from_anchor():
                return  # keep still after a click
            self._settle_until = 0.0
            self.cursor.set_target(*self._target)


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
