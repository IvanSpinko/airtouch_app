"""Command-line entry point: run AirTouch with the OpenCV preview window.

In the L pose (index + thumb straight) the cursor follows the base of the index
finger; bending the thumb clicks / drags, bending the index right-clicks, two fingers
scroll, a held fist pauses, a held shaka opens the on-screen keyboard.
Ctrl+Alt+G instantly disables/enables control.
Press q or Esc in the preview window (or Ctrl+C in the console) to quit.

Run: ``python -m gesture_control.main`` (the tray app is ``python -m gesture_control.app``)
"""

from __future__ import annotations

import argparse
import sys

from gesture_control import config, keyboard, settings
from gesture_control.engine import Engine


def run(log: bool = False) -> None:
    Engine(log=log).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="AirTouch: control the PC with hand gestures")
    parser.add_argument(
        "--log", action="store_true",
        help="save landmarks and mouse actions to data/ for diagnosing false clicks",
    )
    parser.add_argument(
        "--no-admin", action="store_true",
        help="do not ask for administrator rights (on-screen keyboard clicks will not work)",
    )
    # set on the elevated copy: it has its own console window, keep it open at the end
    parser.add_argument("--elevated", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    settings.load()
    if config.RUN_AS_ADMIN and not args.no_admin and not keyboard.is_admin():
        # osk.exe ignores input from normal-rights processes (UIPI), see keyboard.py.
        if keyboard.relaunch_as_admin([*sys.argv[1:], "--elevated"]):
            print("Restarted as administrator in a new window.")
            return
        print("Administrator rights declined: continuing without on-screen keyboard clicks.")
    try:
        run(log=args.log)
    except KeyboardInterrupt:
        pass
    except Exception:
        if not args.elevated:
            raise
        import traceback

        traceback.print_exc()
    if args.elevated:
        input("Press Enter to close this window...")


if __name__ == "__main__":
    main()
