"""Debug preview window drawing: camera frame, hand skeleton, gesture, status,
click-pose bars, scroll anchor and the last mouse action.

Used by the engine when ``SHOW_PREVIEW`` is on; the tray app shows a small
indicator instead (overlay.py).
"""

from __future__ import annotations

import cv2
import numpy as np

from gesture_control import config
from gesture_control.controller import GestureController
from gesture_control.gestures import Gesture, HandFeatures
from gesture_control.mouse import WorkZone
from gesture_control.tracker import Hand, draw_hand, near_edges


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


def draw_click_meters(
    frame: np.ndarray, strength: dict[Gesture, float], held: Gesture,
    left: Gesture, right: Gesture,
) -> None:
    """Click pose bars (L = thumb bent, R = index bent): the pose counts once its
    bar crosses the yellow line (CLASSIFIER_MIN_PROB); the cursor stops above the
    orange one (CURSOR_FREEZE_CLICK_LEVEL)."""
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
        for level, c in ((config.CLASSIFIER_MIN_PROB, config.COLOR_METER_ON),
                         (config.CURSOR_FREEZE_CLICK_LEVEL, config.COLOR_METER_OFF)):
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
    event: str | None = None, keyboard_open: bool = False,
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
        text + ("  |  Keyboard" if keyboard_open else ""),
    ]
    if config.PREVIEW_SHOW_FEATURES and hand is not None and features is not None:
        lines.append(
            f"thumb reach {features.thumb_reach:.2f} (out > {config.THUMB_EXTENDED_RATIO})"
            f"  index-middle gap {features.index_middle_gap:.2f}"
        )
    if meters is not None and hand is not None:
        draw_click_meters(out, *meters)
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
