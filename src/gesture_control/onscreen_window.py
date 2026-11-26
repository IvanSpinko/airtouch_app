"""PyQt6 window of the custom on-screen keyboard (the logic is in onscreen.py).

The window is always on top, has no frame and never takes the keyboard focus
(Qt's WindowDoesNotAcceptFocus plus the Windows WS_EX_NOACTIVATE style), so the
app the user types into stays active. It polls ``OnscreenKeyboard.view()`` on a
timer: the engine thread changes the keyboard, the Qt thread only draws it.
Real mouse clicks on keys work too.

Geometry in onscreen.py is in physical pixels (the engine is DPI-aware); Qt works
in device-independent pixels, so everything is divided by the screen's
devicePixelRatio here.
"""

from __future__ import annotations

import ctypes
import sys

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import QApplication, QWidget

from gesture_control import config
from gesture_control.onscreen import (
    LANG, SHIFT, SUGGEST, OnscreenKeyboard, View, handle_rect,
)

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOPMOST = 0x00000008


def _no_activate(hwnd: int) -> None:
    """Makes a window that never becomes the foreground (focused) window."""
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    get = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
    set_ = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
    get.restype = set_.restype = ctypes.c_ssize_t
    get.argtypes = [ctypes.c_void_p, ctypes.c_int]
    set_.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    style = get(hwnd, GWL_EXSTYLE)
    set_(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOPMOST)


class KeyboardWindow(QWidget):
    def __init__(self, keyboard: OnscreenKeyboard) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.keyboard = keyboard
        self._view = View()
        self._drawn: tuple | None = None
        self._styled = False
        self._drag_from: tuple[float, float] | None = None  # real-mouse drag, physical px
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.sync)
        self.timer.start(config.ONSCREEN_REFRESH_MS)

    def _ratio(self) -> float:
        screen = self.screen() or QApplication.primaryScreen()
        return screen.devicePixelRatio() if screen else 1.0

    def sync(self) -> None:
        """Shows / hides / repaints to match the keyboard state."""
        view = self.keyboard.view()
        self._view = view
        if view.visible and view.area is not None:
            r = self._ratio()
            a = view.area
            self.setGeometry(round(a.x / r), round(a.y / r), round(a.w / r), round(a.h / r))
            if not self.isVisible():
                self.show()
                if not self._styled:
                    _no_activate(int(self.winId()))
                    self._styled = True
        elif self.isVisible():
            self.hide()
        state = (view.version, view.flash, view.handle_active)
        if state != self._drawn:
            self._drawn = state
            self.update()

    # -- drawing ----------------------------------------------------------------------
    def _key_color(self, index: int, kind: str) -> str:
        v = self._view
        if index == v.flash:
            return config.ONSCREEN_KEY_PRESSED
        if index == v.hovered:
            return config.ONSCREEN_KEY_ACTIVE
        if kind == SHIFT and v.shift:
            return config.ONSCREEN_KEY_ON
        if kind == SUGGEST:
            return config.ONSCREEN_SUGGESTION
        if kind == "char":
            return config.ONSCREEN_KEY
        return config.ONSCREEN_KEY_SPECIAL

    def paintEvent(self, _event) -> None:
        v = self._view
        if v.area is None:
            return
        r = self._ratio()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(config.ONSCREEN_BACKGROUND))
        radius = config.ONSCREEN_WINDOW_RADIUS_PX
        p.drawRoundedRect(QRectF(self.rect()), radius, radius)
        # drag handle: a grip bar in the middle of the top strip
        h = handle_rect(v.area)
        gw = self.width() * config.ONSCREEN_HANDLE_GRIP_FRACTION
        gh = config.ONSCREEN_HANDLE_GRIP_PX / r
        grip = QRectF((self.width() - gw) / 2, (h.h / r - gh) / 2, gw, gh)
        p.setBrush(QColor(config.ONSCREEN_KEY_ACTIVE if v.handle_active else config.ONSCREEN_HANDLE))
        p.drawRoundedRect(grip, gh / 2, gh / 2)
        font = QFont()
        for i, key in enumerate(v.keys):
            k = key.rect
            rect = QRectF((k.x - v.area.x) / r, (k.y - v.area.y) / r, k.w / r, k.h / r)
            kr = min(rect.width(), rect.height()) * config.ONSCREEN_RADIUS_FRACTION
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(self._key_color(i, key.kind)))
            p.drawRoundedRect(rect, kr, kr)
            label = v.labels[i] if i < len(v.labels) else ""
            if not label:
                continue
            size = rect.height() * config.ONSCREEN_FONT_FRACTION
            if key.kind != "char" and key.kind != LANG:
                size *= config.ONSCREEN_SPECIAL_FONT_SCALE
            font.setPixelSize(max(1, round(size)))
            font.setBold(key.kind == "char")
            p.setFont(font)
            p.setPen(QColor(config.ONSCREEN_TEXT))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
        p.end()

    # -- real mouse ---------------------------------------------------------------------
    def _physical(self, event) -> tuple[float, float]:
        r = self._ratio()
        pos = event.globalPosition()
        return pos.x() * r, pos.y() * r

    def mousePressEvent(self, event) -> None:
        if self._view.area is None:
            return
        x, y = self._physical(event)
        if self.keyboard.on_handle(x, y):
            self._drag_from = (x, y)
            self.keyboard.set_handle_active(True)
        else:
            self.keyboard.press_at(x, y)
        self.sync()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is None:
            return
        x, y = self._physical(event)
        self.keyboard.move_by(x - self._drag_from[0], y - self._drag_from[1])
        self._drag_from = (x, y)
        self.sync()

    def mouseReleaseEvent(self, _event) -> None:
        if self._drag_from is not None:
            self._drag_from = None
            self.keyboard.set_handle_active(False)
            self.sync()
