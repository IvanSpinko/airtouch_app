"""Cursor control: work-zone mapping, smoothing and pynput output.

The pure mapping logic (``WorkZone``, ``CursorMapper``) has no OS dependencies and is
covered by tests; ``Mouse`` is a thin wrapper around ``pynput``.

Check: ``python -m gesture_control.mouse`` moves the cursor around the screen edges.
"""

from __future__ import annotations

import ctypes
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

from gesture_control import config
from gesture_control.filters import PointFilter


# -- OS helpers -----------------------------------------------------------------
def enable_dpi_awareness() -> None:
    """Make Windows report real pixel sizes, so the cursor reaches every corner
    on scaled (125%, 150%, ...) displays."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(config.DPI_AWARENESS_PER_MONITOR)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


def screen_size() -> tuple[int, int]:
    """Primary monitor size in physical pixels."""
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(config.SM_CXSCREEN), user32.GetSystemMetrics(
            config.SM_CYSCREEN
        )
    raise RuntimeError("screen_size() is only implemented for Windows")


# -- Mapping --------------------------------------------------------------------
@dataclass(frozen=True)
class WorkZone:
    """Rectangle in normalized frame coordinates mapped onto the whole screen."""

    left: float = config.WORK_ZONE_LEFT
    top: float = config.WORK_ZONE_TOP
    right: float = config.WORK_ZONE_RIGHT
    bottom: float = config.WORK_ZONE_BOTTOM
    sensitivity: float = config.CURSOR_SENSITIVITY

    def effective(self) -> tuple[float, float, float, float]:
        """Zone bounds after applying sensitivity (scaled around the center)."""
        cx, cy = (self.left + self.right) / 2, (self.top + self.bottom) / 2
        half_w = (self.right - self.left) / 2 / self.sensitivity
        half_h = (self.bottom - self.top) / 2 / self.sensitivity
        return cx - half_w, cy - half_h, cx + half_w, cy + half_h

    def to_screen(
        self, x: float, y: float, width: int, height: int, clamp: bool = True
    ) -> tuple[float, float]:
        """Maps a normalized frame point to screen pixels (clamped to the screen by default)."""
        left, top, right, bottom = self.effective()
        u = (x - left) / (right - left)
        v = (y - top) / (bottom - top)
        if clamp:
            u = min(max(u, 0.0), 1.0)
            v = min(max(v, 0.0), 1.0)
        return u * (width - 1), v * (height - 1)


def accel_gain(
    speed: float,
    min_gain: float = config.ACCEL_MIN_GAIN,
    max_gain: float = config.ACCEL_MAX_GAIN,
    low: float = config.ACCEL_LOW_SPEED_PX_S,
    high: float = config.ACCEL_HIGH_SPEED_PX_S,
) -> float:
    """Pointer acceleration curve: slow hand -> small gain (precise aiming),
    fast hand -> large gain (quick travel). Smooth S-curve in between."""
    k = min(max((speed - low) / (high - low), 0.0), 1.0)
    k = k * k * (3.0 - 2.0 * k)  # smoothstep
    return min_gain + (max_gain - min_gain) * k


class CursorMapper:
    """Hand point -> smoothed screen position.

    Two modes (``config.CURSOR_MODE``):

    - ``"absolute"``: the work zone is stretched over the screen; a hand position
      always means the same screen position.
    - ``"relative"``: like a mouse — the cursor moves by the change of the hand
      position, scaled by a pointer-acceleration curve. Slow moves are precise, fast
      moves travel far; the hand can be lowered and raised elsewhere without the
      cursor jumping.

    Both modes apply the One Euro Filter and a small dead zone and reset the filter
    when the hand has been lost for a while.
    """

    def __init__(
        self,
        width: int,
        height: int,
        zone: WorkZone | None = None,
        deadzone_px: float = config.CURSOR_DEADZONE_PX,
        lost_reset_s: float = config.HAND_LOST_RESET_S,
        mode: str = config.CURSOR_MODE,
        position_source: Callable[[], tuple[float, float]] | None = None,
    ) -> None:
        if mode not in ("absolute", "relative"):
            raise ValueError(f"unknown cursor mode: {mode!r}")
        self.width = width
        self.height = height
        self.zone = zone or WorkZone()
        self.deadzone_px = deadzone_px
        self.lost_reset_s = lost_reset_s
        self.mode = mode
        self._position_source = position_source
        self._filter = PointFilter()
        self._last_seen: float | None = None
        self._last_out: tuple[int, int] | None = None
        self._prev: tuple[float, float] | None = None
        self._prev_t = 0.0
        self._pos = ((width - 1) / 2, (height - 1) / 2)
        self._sync_position()

    def _sync_position(self) -> None:
        """Relative mode continues from the real cursor (it may have been moved by hand)."""
        if self._position_source is not None:
            x, y = self._position_source()
            self._pos = (float(x), float(y))

    def reset(self) -> None:
        self._filter.reset()
        self._last_out = None
        self._prev = None
        if self.mode == "relative":
            self._sync_position()

    def _relative(self, fx: float, fy: float, t: float) -> tuple[float, float]:
        if self._prev is None or t <= self._prev_t:
            self._prev, self._prev_t = (fx, fy), t
            return self._pos
        dx, dy = fx - self._prev[0], fy - self._prev[1]
        speed = math.hypot(dx, dy) / (t - self._prev_t)
        g = accel_gain(speed)
        x = min(max(self._pos[0] + dx * g, 0.0), self.width - 1.0)
        y = min(max(self._pos[1] + dy * g, 0.0), self.height - 1.0)
        self._pos = (x, y)
        self._prev, self._prev_t = (fx, fy), t
        return self._pos

    def update(self, x: float, y: float, t: float) -> tuple[int, int] | None:
        """Returns a new cursor position, or None if the move is within the dead zone."""
        if self._last_seen is not None and t - self._last_seen > self.lost_reset_s:
            self.reset()
        self._last_seen = t

        relative = self.mode == "relative"
        sx, sy = self.zone.to_screen(x, y, self.width, self.height, clamp=not relative)
        fx, fy = self._filter(sx, sy, t)
        if relative:
            fx, fy = self._relative(fx, fy, t)
        out = (round(fx), round(fy))
        if self._last_out is not None:
            dx, dy = out[0] - self._last_out[0], out[1] - self._last_out[1]
            if (dx * dx + dy * dy) ** 0.5 < self.deadzone_px:
                return None
        self._last_out = out
        return out


def glide_step(
    current: tuple[float, float], target: tuple[float, float], dt: float,
    tau: float = config.CURSOR_GLIDE_TAU_S, snap: float = config.CURSOR_SNAP_PX,
) -> tuple[float, float]:
    """One step of exponential approach from ``current`` to ``target``."""
    dx, dy = target[0] - current[0], target[1] - current[1]
    if (dx * dx + dy * dy) ** 0.5 <= snap:
        return target
    k = 1.0 - math.exp(-dt / tau)
    return current[0] + dx * k, current[1] + dy * k


class CursorAnimator:
    """Background thread that glides the cursor to the latest target at a high rate."""

    def __init__(self, mouse: "Mouse", rate_hz: float = config.CURSOR_ANIMATION_HZ) -> None:
        self._mouse = mouse
        self._period = 1.0 / rate_hz
        self._lock = threading.Lock()
        self._target: tuple[float, float] | None = None
        self._current: tuple[float, float] | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def set_target(self, x: float, y: float) -> None:
        with self._lock:
            if self._target is None:
                # (re)start gliding from wherever the cursor really is now
                self._current = tuple(map(float, self._mouse.position))
            self._target = (x, y)

    def jump_to(self, x: float, y: float) -> None:
        """Moves immediately, without gliding (used right before clicks)."""
        with self._lock:
            self._target = self._current = (float(x), float(y))
        self._mouse.move_to(round(x), round(y))

    def stop_moving(self) -> None:
        """Forgets the target so the physical mouse is not fought with."""
        with self._lock:
            self._target = None

    def start(self) -> "CursorAnimator":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="cursor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def __enter__(self) -> "CursorAnimator":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _loop(self) -> None:
        last = time.perf_counter()
        while self._running:
            time.sleep(self._period)
            now = time.perf_counter()
            dt, last = now - last, now
            with self._lock:
                if self._target is None or self._current is None:
                    continue
                if self._current == self._target:
                    continue
                self._current = glide_step(self._current, self._target, dt)
                x, y = self._current
            self._mouse.move_to(round(x), round(y))


# -- Output ---------------------------------------------------------------------
class Mouse:
    """Sends cursor, button and wheel events to Windows via pynput.

    Buttons are named "left" / "right" so callers do not depend on pynput.
    """

    def __init__(self) -> None:
        from pynput.mouse import Button, Controller

        self._ctl = Controller()
        self._buttons = {"left": Button.left, "right": Button.right}

    @property
    def position(self) -> tuple[int, int]:
        return self._ctl.position

    def move_to(self, x: int, y: int) -> None:
        self._ctl.position = (x, y)

    def press(self, button: str) -> None:
        self._ctl.press(self._buttons[button])

    def release(self, button: str) -> None:
        self._ctl.release(self._buttons[button])

    def click(self, button: str) -> None:
        self._ctl.click(self._buttons[button])

    def scroll(self, ticks: float) -> None:
        """Vertical scroll; fractional ticks give smooth scrolling. Positive = up."""
        self._ctl.scroll(0, ticks)


def _demo() -> None:
    enable_dpi_awareness()
    w, h = screen_size()
    mouse = Mouse()
    print(f"Screen: {w}x{h}. Visiting corners...")
    for x, y in ((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1), (w // 2, h // 2)):
        mouse.move_to(x, y)
        print(f"  -> {mouse.position}")
        time.sleep(config.MOUSE_DEMO_STEP_S)


if __name__ == "__main__":
    _demo()
