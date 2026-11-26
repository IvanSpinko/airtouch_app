"""Controller: decides what to do with the tracked hand.

- ``ControlState`` — global on/off switch flipped by the Ctrl+Alt+G hotkey (from the
  hotkey thread, so it is guarded by a lock).
- ``GestureController`` — state machine that turns gestures into cursor movement,
  clicks, drags, scrolling, pause and the on-screen keyboard toggle.

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


class KeyTarget(Protocol):
    """The custom on-screen keyboard as seen by the controller (onscreen.py)."""

    def hover(self, x: float, y: float) -> tuple[int, int] | None: ...
    def leave(self) -> None: ...
    def press(self) -> bool: ...


class GestureController:
    """Turns recognized gestures into mouse actions.

    - ``CURSOR_GESTURE`` (L pose): the cursor follows the index knuckle. In any other
      pose it stays where it is ("clutch"), like letting go of a mouse.
    - ``LEFT_BUTTON_GESTURE`` (thumb bent): a quick bend clicks when the hand returns
      to the L pose; the button is never held by it, so a false bend cannot drag or
      select. Keeping the thumb bent for ``DRAG_ARM_S`` presses the button ("DRAG
      READY"); moving past ``DRAG_START_PX`` drags; straightening the thumb drops.
    - ``RIGHT_BUTTON_GESTURE`` (index bent): right click on returning to the L pose.
    - A click needs the return to the L pose, so closing the hand into a fist (which
      passes through both bent poses) never clicks; clicks are also ignored for
      ``AFTER_FIST_CLICK_GUARD_S`` after a fist.
    - Two quick clicks land in the same spot -> Windows double click.
    - While a click pose is building up (``click_level`` >= CURSOR_FREEZE_CLICK_LEVEL)
      the cursor stops, so clicks land where aimed.
    - TWO_FINGERS: cursor frozen; scrolls by hand offset from where the pose started
      (``SCROLL_MODE`` "auto", like a pressed mouse wheel) or by hand movement ("drag").
    - FIST held for ``FIST_HOLD_S``: toggles pause.
    - ``KEYBOARD_GESTURE`` (shaka) held for ``KEYBOARD_TOGGLE_HOLD_S``: calls
      ``on_keyboard_toggle`` (opens / closes the on-screen keyboard).
    - Over the custom on-screen keyboard (``keys``) the cursor snaps to the
      highlighted key, a thumb bend presses that key instead of clicking, and
      drag and right click are off.

    OS output goes through ``cursor`` and ``buttons``, so the logic is testable
    without touching the real mouse.
    """

    def __init__(
        self, mapper: CursorMapper, cursor: CursorSink, buttons: ButtonSink,
        on_keyboard_toggle: Callable[[], None] | None = None,
        keys: KeyTarget | None = None,
    ) -> None:
        self.mapper = mapper
        self.cursor = cursor
        self.buttons = buttons
        self.on_keyboard_toggle = on_keyboard_toggle
        self.keys = keys
        self._over_keys = False
        self.paused = False
        # Last mouse action as (label, time), shown briefly in the preview.
        self.last_event: tuple[str, float] | None = None
        self.cursor_gesture = Gesture(config.CURSOR_GESTURE)
        self.left_gesture = Gesture(config.LEFT_BUTTON_GESTURE)
        self.right_gesture = Gesture(config.RIGHT_BUTTON_GESTURE)
        self.keyboard_gesture = Gesture(config.KEYBOARD_GESTURE)
        self._gesture = Gesture.NONE
        self._history: deque[tuple[float, tuple[int, int]]] = deque()
        self._target: tuple[int, int] | None = None
        self._last_seen: float | None = None
        self._click_level = 0.0
        self._left_down = False
        self._left_auto_released = False
        self._dragging = False
        self._bend_anchor: tuple[float, float] = (0.0, 0.0)
        self._bend_since = 0.0
        self._press_since = 0.0
        self._settle_until = 0.0
        self._fist_at = float("-inf")
        self._scrolling = False
        self._scroll_ref_y = 0.0
        self._scroll_last_t = 0.0
        self._scroll_acc = 0.0
        self._fist_since: float | None = None
        self._fist_used = False
        self._kb_since: float | None = None
        self._kb_used = False

    @property
    def gesture(self) -> Gesture:
        return self._gesture

    @property
    def dragging(self) -> bool:
        return self._dragging

    @property
    def scroll_anchor(self) -> float | None:
        """Normalized y of the auto-scroll anchor while auto-scrolling, else None."""
        if self._gesture is Gesture.TWO_FINGERS and config.SCROLL_MODE == "auto":
            return self._scroll_ref_y
        return None

    # -- public API -------------------------------------------------------------
    def release_all(self) -> None:
        """Releases held buttons and stops cursor motion (on disable, quit, lost hand)."""
        if self._left_down:
            self.buttons.release("left")
        self._left_down = False
        self._dragging = False
        self._gesture = Gesture.NONE
        self.cursor.stop_moving()

    def update(
        self, gesture: Gesture, point: tuple[float, float] | None, t: float,
        click_level: float = 0.0,
    ) -> None:
        """Processes one frame. ``point`` is the normalized cursor landmark or None;
        ``click_level`` is how strongly a click pose is shown right now (0..1)."""
        self._click_level = click_level
        if point is None:
            if self._last_seen is not None and t - self._last_seen > config.HAND_LOST_RELEASE_S:
                if self._left_down or self._gesture is not Gesture.NONE:
                    self.release_all()
            return
        self._last_seen = t

        self._update_pause(gesture, t)
        if not self.paused:
            self._update_keyboard(gesture, t)
        if self.paused or gesture is Gesture.FIST:
            if gesture is Gesture.FIST:
                self._fist_at = t
            if self._left_down or self._gesture not in (Gesture.NONE, Gesture.FIST):
                self.release_all()
            self._gesture = gesture
            return

        x, y = point
        raw = self.mapper.zone.to_screen(x, y, self.mapper.width, self.mapper.height)
        new_target = self.mapper.update(x, y, t)  # always, so the filter stays warm
        if new_target is not None:
            self._target = new_target
            self._history.append((t, new_target))
            while self._history and t - self._history[0][0] > config.CURSOR_HISTORY_S:
                self._history.popleft()
        if self._target is None:
            self._target = (round(raw[0]), round(raw[1]))

        prev, self._gesture = self._gesture, gesture
        if gesture is not prev:
            self._exit(prev, gesture, t)
            self._enter(gesture, y, t)
        self._hold(gesture, y, t)

    # -- state machine ----------------------------------------------------------
    def _event(self, label: str, t: float) -> None:
        self.last_event = (label, t)

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
            self._event("PAUSED" if self.paused else "RESUMED", t)

    def _update_keyboard(self, gesture: Gesture, t: float) -> None:
        """Keyboard pose held long enough toggles the on-screen keyboard (once per hold)."""
        if gesture is not self.keyboard_gesture:
            self._kb_since = None
            self._kb_used = False
            return
        if self._kb_since is None:
            self._kb_since = t
        if not self._kb_used and t - self._kb_since >= config.KEYBOARD_TOGGLE_HOLD_S:
            self._kb_used = True
            self._event("KEYBOARD", t)
            if self.on_keyboard_toggle is not None:
                self.on_keyboard_toggle()

    def _anchor(self, t: float) -> tuple[int, int]:
        """Cursor position from CLICK_LOOKBACK_S ago (before the finger started bending)."""
        anchor = self._history[0][1] if self._history else self._target
        for ts, pos in self._history:
            if ts <= t - config.CLICK_LOOKBACK_S:
                anchor = pos
            else:
                break
        return anchor

    def _clicks_allowed(self, t: float) -> bool:
        return t - self._fist_at >= config.AFTER_FIST_CLICK_GUARD_S

    def _enter(self, gesture: Gesture, y: float, t: float) -> None:
        if gesture in (self.left_gesture, self.right_gesture):
            # Repeated click during the settle window reuses the same spot (double click).
            # Over the keyboard the cursor already sits on the snapped key center.
            if t >= self._settle_until and not self._over_keys:
                ax, ay = self._anchor(t)
                self.cursor.jump_to(ax, ay)
            self._bend_anchor = self._target
            self._bend_since = t
            self._dragging = False
            self._left_auto_released = False
        elif gesture is Gesture.TWO_FINGERS:
            self.cursor.stop_moving()
            self._scroll_ref_y = y
            self._scroll_last_t = t
            self._scroll_acc = 0.0
            self._scrolling = False

    def _exit(self, gesture: Gesture, new: Gesture, t: float) -> None:
        returned = (
            new is self.cursor_gesture
            and t - self._bend_since >= config.CLICK_MIN_HOLD_S
            and self._clicks_allowed(t)
        )
        if gesture is self.left_gesture:
            if self._left_down:
                self.buttons.release("left")
                self._left_down = False
                self._event("DROP" if self._dragging else "LEFT UP", t)
                if not self._dragging:
                    self._settle_until = t + config.CLICK_SETTLE_S
                self._dragging = False
            elif returned and not self._left_auto_released:
                if self._over_keys and self.keys is not None:
                    if self.keys.press():
                        self._event("KEY", t)
                else:
                    self.buttons.click("left")
                    self._event("LEFT CLICK", t)
                self._settle_until = t + config.CLICK_SETTLE_S
        elif gesture is self.right_gesture and returned and not self._over_keys:
            self.buttons.click("right")
            self._event("RIGHT CLICK", t)
            self._settle_until = t + config.CLICK_SETTLE_S

    def _moved_from_anchor(self) -> bool:
        # Smoothed position: raw landmarks jitter by several screen pixels per frame.
        dx = self._target[0] - self._bend_anchor[0]
        dy = self._target[1] - self._bend_anchor[1]
        return math.hypot(dx, dy) > config.DRAG_START_PX

    def _hold(self, gesture: Gesture, y: float, t: float) -> None:
        if gesture is self.left_gesture:
            if not self._left_down:
                if (not self._left_auto_released and self._clicks_allowed(t)
                        and not self._over_keys
                        and t - self._bend_since >= config.DRAG_ARM_S):
                    self.buttons.press("left")
                    self._left_down = True
                    self._press_since = t
                    self._event("DRAG READY", t)
                return  # cursor stays put until the drag starts
            if not self._dragging and self._moved_from_anchor():
                self._dragging = True
                self._event("DRAG", t)
            if not self._dragging and t - self._press_since >= config.LEFT_HOLD_STILL_MAX_S:
                self.buttons.release("left")
                self._left_down = False
                self._left_auto_released = True
                self._event("AUTO RELEASE", t)
                return
            if self._dragging:
                self.cursor.set_target(*self._target)
        elif gesture is self.right_gesture:
            pass  # cursor stays put; the click happens on return to the L pose
        elif gesture is Gesture.TWO_FINGERS:
            if config.SCROLL_MODE == "auto":
                self._auto_scroll(y, t)
                return
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
        elif gesture is self.cursor_gesture:
            if self._click_level >= config.CURSOR_FREEZE_CLICK_LEVEL:
                return  # a bend is building up: hold the cursor where it aims
            if t < self._settle_until and not self._moved_from_anchor():
                return  # keep still after a click
            self._settle_until = 0.0
            self.cursor.set_target(*self._snapped(self._target))
        # any other pose: the cursor stays where it is (clutch released)

    def _snapped(self, target: tuple[int, int]) -> tuple[int, int]:
        """Key center when the cursor is over the on-screen keyboard, else ``target``."""
        center = self.keys.hover(*target) if self.keys is not None else None
        self._over_keys = center is not None
        return center if center is not None else target

    def _auto_scroll(self, y: float, t: float) -> None:
        dt, self._scroll_last_t = t - self._scroll_last_t, t
        offset = self._scroll_ref_y - y  # hand above the anchor -> positive
        excess = abs(offset) - config.SCROLL_AUTO_DEADZONE
        if excess <= 0 or dt <= 0:
            self._scrolling = False
            return
        self._scrolling = True
        speed = min(excess * config.SCROLL_AUTO_SPEED, config.SCROLL_AUTO_MAX_TICKS_S)
        ticks = math.copysign(speed * dt, offset)
        if config.SCROLL_INVERT:
            ticks = -ticks
        self._scroll_acc += ticks
        if abs(self._scroll_acc) >= config.SCROLL_MIN_TICKS:
            self.buttons.scroll(self._scroll_acc)
            self._scroll_acc = 0.0


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
