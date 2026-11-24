"""Entry point: camera -> hand tracking -> smoothed cursor.

Stage 1: the cursor follows the index fingertip inside the work zone;
Ctrl+Alt+G instantly disables/enables control. Press q or Esc in the preview
window (or Ctrl+C in the console) to quit.

Run: ``python -m gesture_control.main``
"""

from __future__ import annotations

import cv2
import numpy as np

from gesture_control import config
from gesture_control.camera import Camera, FpsCounter
from gesture_control.controller import ControlState, HotkeyListener
from gesture_control.mouse import (
    CursorAnimator, CursorMapper, Mouse, WorkZone, enable_dpi_awareness, screen_size,
)
from gesture_control.tracker import Hand, HandTracker, draw_hand


def draw_preview(
    frame: np.ndarray, hand: Hand | None, zone: WorkZone, enabled: bool, fps: float
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    left, top, right, bottom = zone.effective()
    cv2.rectangle(
        out, (int(left * w), int(top * h)), (int(right * w), int(bottom * h)),
        config.COLOR_ZONE, config.PREVIEW_LINE_THICKNESS,
    )
    if hand is not None:
        draw_hand(out, hand)
        x, y = hand.point(config.CURSOR_LANDMARK)
        cv2.circle(
            out, (int(x * w), int(y * h)), config.PREVIEW_CURSOR_RADIUS,
            config.COLOR_CURSOR_POINT, cv2.FILLED,
        )
    lines = [
        f"FPS: {fps:.1f}",
        f"Hand: {hand.handedness if hand else '-'}",
        "Control: ON" if enabled else f"Control: OFF ({config.HOTKEY_TOGGLE})",
    ]
    color = config.COLOR_TEXT_ACTIVE if enabled else config.COLOR_TEXT_PAUSED
    ox, oy = config.PREVIEW_TEXT_ORIGIN
    for i, text in enumerate(lines):
        cv2.putText(
            out, text, (ox, oy + i * config.PREVIEW_TEXT_LINE_HEIGHT),
            cv2.FONT_HERSHEY_SIMPLEX, config.PREVIEW_FONT_SCALE, color,
            config.PREVIEW_FONT_THICKNESS,
        )
    return out


def run() -> None:
    enable_dpi_awareness()
    width, height = screen_size()
    zone = WorkZone()
    mapper = CursorMapper(width, height, zone)
    animator = CursorAnimator(Mouse())
    state = ControlState()
    state.subscribe(lambda on: print("Control", "enabled" if on else "disabled"))
    # Re-enabling must not glide the cursor in from where it was disabled.
    state.subscribe(lambda on: mapper.reset() if on else animator.stop_moving())
    fps = FpsCounter()

    print(f"Screen {width}x{height}. {config.HOTKEY_TOGGLE} toggles control, q/Esc quits.")
    with Camera() as cam, HandTracker() as tracker, HotkeyListener(state), animator:
        while cam.is_running:
            item = cam.read(timeout=1.0)
            if item is None:
                continue
            frame, ts = item
            hand = tracker.process(frame, ts)
            enabled = state.enabled

            if hand is not None and enabled:
                x, y = hand.point(config.CURSOR_LANDMARK)
                pos = mapper.update(x, y, ts)
                if pos is not None:
                    animator.set_target(*pos)

            current_fps = fps.tick()
            if config.SHOW_PREVIEW:
                cv2.imshow(
                    config.PREVIEW_WINDOW_NAME,
                    draw_preview(frame, hand, zone, enabled, current_fps),
                )
                if cv2.waitKey(config.PREVIEW_WAIT_KEY_MS) & 0xFF in config.PREVIEW_QUIT_KEYS:
                    break
    cv2.destroyAllWindows()


def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
