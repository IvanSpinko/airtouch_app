"""Qt widgets rendered off-screen (no window appears)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from gesture_control import config  # noqa: E402
from gesture_control.engine import Snapshot  # noqa: E402
from gesture_control.overlay import Indicator, make_pixmap, status_of  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("snap,text,color", [
    (Snapshot(running=True), "ON", config.STATUS_COLOR_ON),
    (Snapshot(running=True, paused=True), "PAUSED", config.STATUS_COLOR_PAUSED),
    (Snapshot(running=True, enabled=False), "OFF", config.STATUS_COLOR_OFF),
    (Snapshot(running=False), "starting...", config.STATUS_COLOR_OFF),
    (Snapshot(running=False, error="boom"), "ERROR", config.STATUS_COLOR_ERROR),
])
def test_status_of(snap, text, color):
    assert status_of(snap) == (text, color)


def test_icon_has_status_color_in_the_middle(qapp):
    img = make_pixmap(config.STATUS_COLOR_ON).toImage()
    corner = img.pixelColor(0, 0)
    assert corner.alpha() == 0  # round icon, transparent corners
    edge = img.pixelColor(config.ICON_SIZE // 2, 4)
    assert edge.name() == config.STATUS_COLOR_ON


def test_indicator_renders_every_state(qapp):
    ind = Indicator()
    for snap in (
        Snapshot(running=True, hand_visible=True, gesture="l_pose"),
        Snapshot(running=True, hand_visible=True, gesture="thumb_bent", event="LEFT CLICK"),
        Snapshot(running=True, hand_visible=False, enabled=False),
        Snapshot(running=True, hand_visible=True, gesture="shaka", keyboard_open=True),
        Snapshot(running=False, error="Cannot open camera #0"),
    ):
        ind.show_snapshot(snap)
        assert not ind.grab().isNull()
    assert ind.width() == config.INDICATOR_WIDTH


# -- custom on-screen keyboard window ---------------------------------------------------
def test_keyboard_window_follows_keyboard_state_and_paints(qapp):
    from gesture_control.onscreen import OnscreenKeyboard, Suggester
    from gesture_control.onscreen_window import KeyboardWindow

    keys = OnscreenKeyboard(sender=object(), suggesters={"en": Suggester([]), "ru": Suggester([])})
    keys.set_screen(1920, 1080)
    win = KeyboardWindow(keys)
    win.sync()
    assert not win.isVisible()
    keys.set_visible(True)
    win.sync()
    assert win.isVisible() and win.width() > 0
    assert not win.isActiveWindow()
    win.grab()  # paints every key without errors
    keys.set_visible(False)
    win.sync()
    assert not win.isVisible()
