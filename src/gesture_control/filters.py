"""One Euro Filter — adaptive cursor smoothing.

Casiez, Roussel, Vogel. "1€ Filter: A Simple Speed-based Low-pass Filter for
Noisy Input in Interactive Systems", CHI 2012.

At rest the cutoff frequency is low (strong jitter smoothing); during fast movement
it grows with speed (low lag).

Check: ``python -m gesture_control.filters`` prints a noisy and a smoothed signal.
"""

from __future__ import annotations

import math

from gesture_control import config


def _smoothing_factor(dt: float, cutoff: float) -> float:
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)


class OneEuroFilter:
    """Filter for a single scalar signal."""

    def __init__(
        self,
        min_cutoff: float = config.FILTER_MIN_CUTOFF,
        beta: float = config.FILTER_BETA,
        d_cutoff: float = config.FILTER_D_CUTOFF,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t = x, t
            return x
        dt = t - self._t
        if dt <= 0.0:
            return self._x
        a_d = _smoothing_factor(dt, self.d_cutoff)
        dx = (x - self._x) / dt
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _smoothing_factor(dt, cutoff)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x


class PointFilter:
    """One Euro Filter for a 2D point (x, y)."""

    def __init__(self, **kwargs) -> None:
        self._fx = OneEuroFilter(**kwargs)
        self._fy = OneEuroFilter(**kwargs)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self._fx(x, t), self._fy(y, t)


def _demo() -> None:
    import random

    f = OneEuroFilter()
    rate = config.CAMERA_FPS
    for i in range(rate):
        t = i / rate
        noisy = 500.0 + random.gauss(0.0, 5.0)
        print(f"t={t:.2f}  raw={noisy:7.2f}  filtered={f(noisy, t):7.2f}")


if __name__ == "__main__":
    _demo()
