"""Entry point: camera -> hand tracking -> gestures -> mouse.

The cursor follows the base of the index finger; pinches click and drag,
two fingers scroll, a held fist pauses. Ctrl+Alt+G instantly disables/enables control.
Press q or Esc in the preview window (or Ctrl+C in the console) to quit.

Run: ``python -m gesture_control.main``
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from gesture_control import config
from gesture_control.camera import Camera, FpsCounter
from gesture_control.controller import ControlState, GestureController, HotkeyListener
from gesture_control.gestures import Gesture, GestureClassifier, GestureDetector, HandFeatures
from gesture_control.mouse import (
    CursorAnimator, CursorMapper, Mouse, WorkZone, enable_dpi_awareness, screen_size,
)
from gesture_control.tracker import Hand, HandTracker, draw_hand, near_edges


def recent_event(ctl: GestureController, now: float) -> str | None:
    if ctl.last_event and now - ctl.last_event[1] <= config.PREVIEW_EVENT_FLASH_S:
        return ctl.last_event[0]
    return None


def status_line(enabled: bool, ctl: GestureController) -> tuple[str, tuple[int, int, int]]:
    if not enabled:
        return f"Control: OFF ({config.HOTKEY_TOGGLE})", config.COLOR_TEXT_PAUSED
    if ctl.paused:
        return "Control: PAUSED (hold fist)", config.COLOR_TEXT_PAUSED
    return "Control: ON", config.COLOR_TEXT_ACTIVE


def draw_edge_warning(frame: np.ndarray, sides: set[str]) -> None:
    """Red bar along each frame edge the hand is about to leave."""
    h, w = frame.shape[:2]
    t = config.EDGE_WARN_THICKNESS
    bars = {
        "left": ((0, 0), (t, h - 1)),
        "right": ((w - 1 - t, 0), (w - 1, h - 1)),
        "top": ((0, 0), (w - 1, t)),
        "bottom": ((0, h - 1 - t), (w - 1, h - 1)),
    }
    for side in sides:
        cv2.rectangle(frame, *bars[side], config.COLOR_EDGE_WARN, cv2.FILLED)


def draw_pinch_meters(
    frame: np.ndarray, strength: dict[Gesture, float], held: Gesture,
    left: Gesture, right: Gesture,
) -> None:
    """Pinch strength bars: the button goes down when a bar crosses the yellow
    line (PINCH_STRENGTH_ON) and up when it falls below the orange one (OFF)."""
    h = frame.shape[0]
    w, bh, gap, m = (config.PREVIEW_METER_WIDTH, config.PREVIEW_METER_HEIGHT,
                     config.PREVIEW_METER_GAP, config.PREVIEW_METER_MARGIN)
    label_w = 2 * m + 10
    for row, (name, g) in enumerate((("L", left), ("R", right))):
        y = h - m - (2 - row) * bh - (1 - row) * gap
        x = label_w
        cv2.putText(frame, name, (m, y + bh - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    config.PREVIEW_FONT_SCALE, config.COLOR_BONE, config.PREVIEW_FONT_THICKNESS)
        cv2.rectangle(frame, (x, y), (x + w, y + bh), config.COLOR_METER_BG, cv2.FILLED)
        fill = int(w * min(max(strength.get(g, 0.0), 0.0), 1.0))
        color = config.COLOR_METER_ACTIVE if held is g else config.COLOR_METER_FILL
        cv2.rectangle(frame, (x, y), (x + fill, y + bh), color, cv2.FILLED)
        for level, c in ((config.PINCH_STRENGTH_ON, config.COLOR_METER_ON),
                         (config.PINCH_STRENGTH_OFF, config.COLOR_METER_OFF)):
            lx = x + int(w * level)
            cv2.line(frame, (lx, y - 2), (lx, y + bh + 2), c, 2)


def draw_event(frame: np.ndarray, text: str) -> None:
    """Big flash of the last mouse action at the top center of the preview."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                  config.PREVIEW_EVENT_FONT_SCALE, config.PREVIEW_EVENT_THICKNESS)
    x = (frame.shape[1] - tw) // 2
    y = th + config.PREVIEW_METER_MARGIN
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, config.PREVIEW_EVENT_FONT_SCALE,
                config.COLOR_EVENT, config.PREVIEW_EVENT_THICKNESS)


def draw_preview(
    frame: np.ndarray, hand: Hand | None, zone: WorkZone, gesture: Gesture,
    status: tuple[str, tuple[int, int, int]], fps: float, scroll_anchor: float | None = None,
    model_conf: float | None = None, features: HandFeatures | None = None,
    meters: tuple[dict[Gesture, float], Gesture, Gesture, Gesture] | None = None,
    event: str | None = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    if config.CURSOR_MODE == "absolute":
        # Only meaningful in absolute mode, where this rectangle is the whole screen.
        left, top, right, bottom = zone.effective()
        cv2.rectangle(
            out, (int(left * w), int(top * h)), (int(right * w), int(bottom * h)),
            config.COLOR_ZONE, config.PREVIEW_LINE_THICKNESS,
        )
    if hand is not None:
        draw_edge_warning(out, near_edges(hand))
    if scroll_anchor is not None:
        # Auto scroll: anchor line and the dead zone around it.
        ay = int(scroll_anchor * h)
        dz = int(config.SCROLL_AUTO_DEADZONE * h)
        cv2.rectangle(out, (0, ay - dz), (w - 1, ay + dz), config.COLOR_SCROLL_DEADZONE,
                      config.PREVIEW_LINE_THICKNESS)
        cv2.line(out, (0, ay), (w - 1, ay), config.COLOR_SCROLL_ANCHOR,
                 config.PREVIEW_LINE_THICKNESS)
    if hand is not None:
        draw_hand(out, hand)
        x, y = hand.cursor_point()
        cv2.circle(
            out, (int(x * w), int(y * h)), config.PREVIEW_CURSOR_RADIUS,
            config.COLOR_CURSOR_POINT, cv2.FILLED,
        )
    text, color = status
    lines = [
        f"FPS: {fps:.1f}",
        f"Hand: {hand.handedness if hand else '-'}",
        f"Gesture: {gesture.value}" + (f" ({model_conf:.0%})" if model_conf is not None else ""),
        text,
    ]
    if config.PREVIEW_SHOW_PINCH_DEBUG and hand is not None and features is not None:
        # Pinch distances vs the click threshold: below it = pinched.
        lines.append(
            f"thumb-index {features.pinch_index:.2f}  thumb-middle {features.pinch_middle:.2f}"
        )
    if meters is not None and hand is not None:
        draw_pinch_meters(out, *meters)
    if event:
        draw_event(out, event)
    ox, oy = config.PREVIEW_TEXT_ORIGIN
    for i, line in enumerate(lines):
        cv2.putText(
            out, line, (ox, oy + i * config.PREVIEW_TEXT_LINE_HEIGHT),
            cv2.FONT_HERSHEY_SIMPLEX, config.PREVIEW_FONT_SCALE, color,
            config.PREVIEW_FONT_THICKNESS,
        )
    return out


def run(log: bool = False) -> None:
    enable_dpi_awareness()
    width, height = screen_size()
    zone = WorkZone()
    mouse = Mouse()
    # Relative mode continues from wherever the real cursor is.
    mapper = CursorMapper(width, height, zone, position_source=lambda: mouse.position)
    animator = CursorAnimator(mouse)
    classifier = GestureClassifier.load_if_available()
    detector = GestureDetector(classifier)
    ctl = GestureController(mapper, animator, mouse)
    state = ControlState()
    state.subscribe(lambda on: print("Control", "enabled" if on else "disabled"))
    fps = FpsCounter()
    logger = None
    if log:
        from gesture_control.sessionlog import SessionLogger

        logger = SessionLogger()
        print(f"Logging landmarks and actions to {logger.path} (press m to mark a mistake)")

    print(f"Screen {width}x{height}. {config.HOTKEY_TOGGLE} toggles control, q/Esc quits.")
    print("Gestures:", "trained classifier + rules" if classifier else "rules only (no model yet)")
    was_enabled = True
    gesture = Gesture.NONE
    with Camera() as cam, HandTracker() as tracker, HotkeyListener(state), animator:
        try:
            while cam.is_running:
                item = cam.read(timeout=1.0)
                if item is None:
                    continue
                frame, ts = item
                hand = tracker.process(frame, ts)
                enabled = state.enabled

                if enabled:
                    if not was_enabled:
                        # Re-enabling must not glide the cursor in from a stale spot.
                        mapper.reset()
                    gesture = detector.update(
                        hand.landmarks if hand else None,
                        hand.handedness if hand else "Right",
                    )
                    point = hand.cursor_point() if hand else None
                    ctl.update(gesture, point, ts, max(detector.strength.values()))
                elif was_enabled:
                    ctl.release_all()
                    detector.reset()
                    gesture = Gesture.NONE
                was_enabled = enabled
                if logger is not None:
                    logger.write(
                        ts, hand.landmarks if hand else None, hand.handedness if hand else "",
                        gesture.value, detector.strength[ctl.left_gesture],
                        detector.strength[ctl.right_gesture], ctl.last_event,
                    )

                current_fps = fps.tick()
                if config.SHOW_PREVIEW:
                    cv2.imshow(
                        config.PREVIEW_WINDOW_NAME,
                        draw_preview(
                            frame, hand, zone, gesture, status_line(enabled, ctl), current_fps,
                            ctl.scroll_anchor,
                            max(detector.probs.values()) if detector.probs and hand else None,
                            detector.features,
                            (detector.strength, gesture, ctl.left_gesture, ctl.right_gesture),
                            recent_event(ctl, ts),
                        ),
                    )
                    key = cv2.waitKey(config.PREVIEW_WAIT_KEY_MS) & 0xFF
                    if key in config.PREVIEW_QUIT_KEYS:
                        break
                    if key == config.SESSION_MARK_KEY and logger is not None:
                        logger.mark(ts)
                        ctl.last_event = ("MARKED", ts)
        finally:
            # Never leave a mouse button stuck down.
            ctl.release_all()
            if logger is not None:
                logger.close()
                print(f"Session log saved: {logger.path}")
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="AirTouch: control the PC with hand gestures")
    parser.add_argument(
        "--log", action="store_true",
        help="save landmarks and mouse actions to data/ for diagnosing false clicks",
    )
    args = parser.parse_args()
    try:
        run(log=args.log)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
