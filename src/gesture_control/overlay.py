"""PyQt6 indicator: a small always-on-top window and the tray icon images.

The indicator sits in a screen corner and shows whether the camera sees the hand,
which gesture is recognized and the control status. It ignores the mouse
(clicks go through it) and never takes focus.

Check: ``python -m gesture_control.overlay`` shows the indicator with demo states.
"""

from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget

from gesture_control import config

# Human-readable gesture names for the indicator.
GESTURE_LABELS = {
    "none": "-",
    "l_pose": "L: cursor",
    "thumb_bent": "thumb bent: left",
    "index_bent": "index bent: right",
    "two_fingers": "scroll",
    "fist": "fist",
    "open_palm": "open palm",
    "shaka": "shaka: keyboard",
}


def status_of(snapshot) -> tuple[str, str]:
    """(text, color) of the control status for a Snapshot."""
    if snapshot.error:
        return "ERROR", config.STATUS_COLOR_ERROR
    if not snapshot.running:
        return "starting...", config.STATUS_COLOR_OFF
    if not snapshot.enabled:
        return "OFF", config.STATUS_COLOR_OFF
    if snapshot.paused:
        return "PAUSED", config.STATUS_COLOR_PAUSED
    return "ON", config.STATUS_COLOR_ON


def make_icon(color: str, size: int = config.ICON_SIZE) -> QIcon:
    """Round app icon in the status color with the letter A."""
    return QIcon(make_pixmap(color, size))


def make_pixmap(color: str, size: int = config.ICON_SIZE) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QBrush(QColor(color)))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QRectF(1, 1, size - 2, size - 2))
    p.setPen(QColor(config.ICON_LETTER_COLOR))
    font = QFont()
    font.setBold(True)
    font.setPixelSize(int(size * config.ICON_LETTER_SCALE))
    p.setFont(font)
    p.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, "A")
    p.end()
    return pm


class Indicator(QWidget):
    """Small translucent always-on-top status window."""

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool
                         | Qt.WindowType.WindowTransparentForInput
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.resize(config.INDICATOR_WIDTH, config.INDICATOR_HEIGHT)
        self._hand = False
        self._gesture = "-"
        self._status, self._status_color = "starting...", config.STATUS_COLOR_OFF
        self._extra = ""
        self.move_to_corner()

    def move_to_corner(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        m = config.INDICATOR_MARGIN
        self.move(screen.right() - self.width() - m, screen.top() + m)

    def show_snapshot(self, snapshot) -> None:
        self._hand = snapshot.hand_visible
        self._gesture = GESTURE_LABELS.get(snapshot.gesture, snapshot.gesture) if self._hand else "no hand"
        self._status, self._status_color = status_of(snapshot)
        extras = []
        if snapshot.keyboard_open:
            extras.append("keyboard")
        if snapshot.event:
            extras.append(snapshot.event)
        self._extra = "  ".join(extras)
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setBrush(QColor(config.INDICATOR_BACKGROUND))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, config.INDICATOR_RADIUS, config.INDICATOR_RADIUS)
        # hand dot
        d = config.INDICATOR_DOT
        p.setBrush(QColor(config.STATUS_COLOR_ON if self._hand else config.STATUS_COLOR_OFF))
        p.drawEllipse(QRectF(config.INDICATOR_PAD, (self.height() - d) / 2, d, d))
        x = config.INDICATOR_PAD * 2 + d
        font = QFont()
        font.setPixelSize(config.INDICATOR_FONT_PX)
        p.setFont(font)
        p.setPen(QColor(config.INDICATOR_TEXT))
        p.drawText(QRectF(x, 4, self.width() - x, self.height() / 2),
                   Qt.AlignmentFlag.AlignVCenter, self._gesture)
        p.setPen(QColor(self._status_color))
        line2 = self._status + (f"   {self._extra}" if self._extra else "")
        p.drawText(QRectF(x, self.height() / 2 - 2, self.width() - x, self.height() / 2),
                   Qt.AlignmentFlag.AlignVCenter, line2)
        p.end()


def _demo() -> None:
    import itertools
    import sys

    from gesture_control.engine import Snapshot

    app = QApplication(sys.argv)
    ind = Indicator()
    ind.show()
    states = itertools.cycle([
        Snapshot(running=True, hand_visible=True, gesture="l_pose"),
        Snapshot(running=True, hand_visible=True, gesture="thumb_bent", event="LEFT CLICK"),
        Snapshot(running=True, hand_visible=True, gesture="fist", paused=True),
        Snapshot(running=True, hand_visible=False, enabled=False),
        Snapshot(running=True, hand_visible=True, gesture="shaka", keyboard_open=True),
    ])
    timer = QTimer()
    timer.timeout.connect(lambda: ind.show_snapshot(next(states)))
    timer.start(config.OVERLAY_DEMO_STEP_MS)
    sys.exit(app.exec())


if __name__ == "__main__":
    _demo()
