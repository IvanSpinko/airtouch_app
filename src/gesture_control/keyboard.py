"""Windows on-screen keyboard (osk.exe): open, close, size, admin rights.

osk.exe runs with high integrity (UIAccess). Windows blocks input and window
messages from a normal-rights process to it (UIPI), so AirTouch needs to run as
administrator to click its keys and to close it. ``relaunch_as_admin`` restarts
the app elevated (UAC prompt).

Before launch the keyboard is configured through its registry settings: an
enlarged window at the bottom of the screen and word prediction on. Layouts and
language switching come from Windows.

Check: ``python -m gesture_control.keyboard --check`` (run from an administrator
terminal) opens the keyboard, clicks its close button with pynput and reports
whether the click got through. Without arguments it toggles the keyboard.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import subprocess
import sys
import time
import winreg
from dataclasses import dataclass

from gesture_control import config


# -- admin rights -----------------------------------------------------------------
def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return False


def relaunch_as_admin(args: list[str]) -> bool:
    """Starts ``python -m gesture_control.main <args>`` elevated (UAC prompt).

    Returns True if the elevated copy was started (the caller should exit),
    False if the user declined the prompt or it failed.
    """
    params = subprocess.list2cmdline(["-m", "gesture_control.main", *args])
    result = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, params, str(config.PROJECT_ROOT), config.SW_SHOWNORMAL
    )
    return int(result) > config.SHELL_EXECUTE_OK


# -- geometry -------------------------------------------------------------------------
@dataclass(frozen=True)
class OskGeometry:
    """Window position and size in the units osk.exe stores (96-DPI pixels)."""

    left: int
    top: int
    width: int
    height: int


def enlarged_geometry(screen_w: int, screen_h: int, scale: float) -> OskGeometry:
    """Wide keyboard docked to the bottom of the screen, easy to hit with a pinch.

    ``screen_w``/``screen_h`` are physical pixels, ``scale`` is DPI / 96.
    """
    w_dip, h_dip = screen_w / scale, screen_h / scale
    width = round(w_dip * config.OSK_WIDTH_FRACTION)
    height = round(h_dip * config.OSK_HEIGHT_FRACTION)
    left = round((w_dip - width) / 2)
    top = round(h_dip - height - h_dip * config.OSK_BOTTOM_MARGIN_FRACTION)
    return OskGeometry(left, top, width, height)


def configure(geometry: OskGeometry, word_prediction: bool = config.OSK_WORD_PREDICTION) -> None:
    """Writes the keyboard settings osk.exe reads when it starts."""
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, config.OSK_REGISTRY_KEY) as key:
        for name, value in (
            ("WindowLeft", geometry.left), ("WindowTop", geometry.top),
            ("WindowWidth", geometry.width), ("WindowHeight", geometry.height),
            ("UseTextPrediction", int(word_prediction)),
            # docked mode would ignore the size; "clicking" mode, not hover/scan
            ("Dock", 0), ("Mode", config.OSK_MODE_CLICK),
        ):
            winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)


# -- window control -------------------------------------------------------------------
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.FindWindowW.restype = wt.HWND
_user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]


def window() -> int | None:
    return _user32.FindWindowW(config.OSK_WINDOW_CLASS, None) or None


def is_open() -> bool:
    return window() is not None


def open_keyboard() -> bool:
    """Configures and starts osk.exe; returns True once its window exists."""
    if is_open():
        return True
    from gesture_control.mouse import enable_dpi_awareness, screen_size

    enable_dpi_awareness()
    w, h = screen_size()
    configure(enlarged_geometry(w, h, _user32.GetDpiForSystem() / config.BASE_DPI))
    # ShellExecute, not CreateProcess: osk.exe is a UIAccess app and CreateProcess
    # refuses it with "requires elevation".
    ctypes.windll.shell32.ShellExecuteW(None, "open", config.OSK_PATH, None, None,
                                        config.SW_SHOWNORMAL)
    return _wait(lambda: is_open())


def close_keyboard() -> bool:
    """Closes osk.exe (needs admin rights); returns True once it is gone."""
    hwnd = window()
    if hwnd is None:
        return True
    if not _user32.PostMessageW(hwnd, config.WM_CLOSE, 0, 0):
        return False  # blocked by UIPI: not running as administrator
    return _wait(lambda: not is_open())


def toggle() -> bool:
    """Opens or closes the keyboard; returns True if it is open afterwards."""
    if is_open():
        close_keyboard()
    else:
        open_keyboard()
    return is_open()


def _wait(done, timeout: float = config.OSK_WAIT_S) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if done():
            return True
        time.sleep(config.OSK_POLL_S)
    return done()


# -- check (stage 4, step 1) ----------------------------------------------------------
def check_clicks() -> bool:
    """Do pynput clicks reach osk.exe? Clicks its close button and sees if it closes."""
    from pynput.mouse import Button, Controller

    print(f"running as administrator: {is_admin()}")
    print("Do not touch the mouse or the keyboard window during the check.")
    if not open_keyboard():
        print("could not open osk.exe")
        return False
    time.sleep(config.OSK_CHECK_SETTLE_S)
    if not is_open():
        print("INVALID: the keyboard closed before the test click (closed by hand?). Run again.")
        return False
    rect = wt.RECT()
    _user32.GetWindowRect(window(), ctypes.byref(rect))
    mouse = Controller()
    old = mouse.position
    mouse.position = (rect.right - config.OSK_CLOSE_BUTTON_DX, rect.top + config.OSK_CLOSE_BUTTON_DY)
    time.sleep(config.OSK_CHECK_SETTLE_S)
    mouse.click(Button.left)
    # Short wait: only a closure caused by our click counts.
    closed = _wait(lambda: not is_open(), timeout=config.OSK_CHECK_RESULT_S)
    mouse.position = old
    if closed:
        print("OK: pynput clicks reach osk.exe")
    else:
        print("BLOCKED: osk.exe ignored the click (run as administrator). Close it by hand.")
    return closed


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows on-screen keyboard control")
    parser.add_argument("--check", action="store_true", help="test that pynput clicks reach osk.exe")
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if check_clicks() else 1)
    print("keyboard open:", toggle())


if __name__ == "__main__":
    main()
