"""Record hand landmarks for training the gesture classifier.

Run from the project root: ``python -m ml.record``

Keys in the preview window:
  1..7   choose the gesture to record (list shown on screen)
  Space  start / stop recording the chosen gesture
  u      undo the last few recorded samples
  q/Esc  quit

While recording, every frame with a hand is saved to ``data/gestures.csv`` —
landmark coordinates only, never images. Vary hand (left / right), distance,
angle and lighting while recording; for "none" record relaxed hands, transitions
and anything that is not one of the other gestures.
"""

from __future__ import annotations

import time
import uuid

import cv2

from gesture_control import config
from gesture_control.camera import Camera
from gesture_control.tracker import HandTracker, draw_hand
from ml.dataset import append_rows, load, remove_last_rows


def draw_ui(frame, label: str, recording: bool, counts: dict[str, int]) -> None:
    ox, oy = config.PREVIEW_TEXT_ORIGIN
    line = config.PREVIEW_TEXT_LINE_HEIGHT
    state = "REC" if recording else "paused (Space)"
    color = config.COLOR_TEXT_PAUSED if recording else config.COLOR_TEXT_ACTIVE
    cv2.putText(frame, f"{label}  [{state}]", (ox, oy), cv2.FONT_HERSHEY_SIMPLEX,
                config.PREVIEW_FONT_SCALE, color, config.PREVIEW_FONT_THICKNESS)
    for i, name in enumerate(config.GESTURE_CLASSES):
        n = counts.get(name, 0)
        done = n >= config.RECORD_TARGET_PER_CLASS
        text = f"{i + 1} {name}: {n}/{config.RECORD_TARGET_PER_CLASS}"
        c = config.COLOR_TEXT_ACTIVE if done else config.COLOR_BONE
        if name == label:
            text = "> " + text
        cv2.putText(frame, text, (ox, oy + (i + 1) * line), cv2.FONT_HERSHEY_SIMPLEX,
                    config.PREVIEW_FONT_SCALE, c, config.PREVIEW_FONT_THICKNESS)


def main() -> None:
    path = config.GESTURE_DATA_PATH
    counts = load(path).counts()
    session = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    label = config.GESTURE_CLASSES[0]
    recording = False
    history: list[str] = []  # labels of rows written this session, for undo

    print(__doc__)
    with Camera() as cam, HandTracker() as tracker:
        while cam.is_running:
            item = cam.read(timeout=1.0)
            if item is None:
                continue
            frame, ts = item
            hand = tracker.process(frame, ts)
            view = frame.copy()
            if hand is not None:
                draw_hand(view, hand)
                if recording:
                    append_rows(path, [(label, hand.handedness, session, hand.landmarks)])
                    counts[label] = counts.get(label, 0) + 1
                    history.append(label)
            draw_ui(view, label, recording, counts)
            cv2.imshow("record gestures", view)

            key = cv2.waitKey(config.PREVIEW_WAIT_KEY_MS) & 0xFF
            if key in config.PREVIEW_QUIT_KEYS:
                break
            if key in config.RECORD_KEYS:
                label, recording = config.RECORD_KEYS[key], False
            elif key == config.RECORD_TOGGLE_KEY:
                recording = not recording
            elif key == config.RECORD_UNDO_KEY and history:
                n = remove_last_rows(path, min(config.RECORD_UNDO_COUNT, len(history)))
                if n:  # history[-0:] would be the whole list
                    for name in history[-n:]:
                        counts[name] -= 1
                    del history[-n:]
                recording = False
    cv2.destroyAllWindows()
    print("Samples per gesture:", counts)


if __name__ == "__main__":
    main()
