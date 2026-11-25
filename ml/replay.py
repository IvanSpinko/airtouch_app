"""Replay a session log through the current gesture code.

Run from the project root: ``python -m ml.replay [data/session-....csv]``
(defaults to the newest log). Record one with ``python -m gesture_control.main --log``
and press ``m`` in the preview right after a false click.

Prints the mouse actions the current code produces for the recorded hand movement
(no mouse is touched), and for every MARK a frame-by-frame trace of the moments
before it - gesture, click-pose strengths and what the model saw - to see what caused it.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gesture_control import config
from gesture_control.controller import GestureController
from gesture_control.gestures import Gesture, GestureClassifier, GestureDetector
from gesture_control.mouse import CursorMapper
from gesture_control.sessionlog import COORDS


@dataclass
class Frame:
    t: float
    landmarks: np.ndarray | None
    handedness: str
    logged_event: str


class _NullCursor:
    def set_target(self, x, y): ...
    def jump_to(self, x, y): ...
    def stop_moving(self): ...


class _NullButtons:
    def press(self, b): ...
    def release(self, b): ...
    def click(self, b): ...
    def scroll(self, ticks): ...


def load_log(path: Path) -> tuple[list[Frame], list[float]]:
    frames, marks = [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = float(row["t"])
            if row["event"] == "MARK":
                marks.append(t)
                continue
            lm = None
            if row[COORDS[0]]:
                lm = np.array([float(row[c]) for c in COORDS], np.float32).reshape(-1, 3)
            frames.append(Frame(t, lm, row["handedness"] or "Right", row["event"]))
    return frames, marks


def cursor_point(lm: np.ndarray) -> tuple[float, float]:
    w = config.CURSOR_POINT_WEIGHTS
    total = sum(w.values())
    return (
        float(sum(lm[i, 0] * v for i, v in w.items()) / total),
        float(sum(lm[i, 1] * v for i, v in w.items()) / total),
    )


def replay(frames: list[Frame], classifier: GestureClassifier | None):
    """Runs detector + controller; returns per-frame trace rows."""
    detector = GestureDetector(classifier)
    ctl = GestureController(CursorMapper(1920, 1080), _NullCursor(), _NullButtons())
    trace, last_event = [], None
    for fr in frames:
        gesture = detector.update(fr.landmarks, fr.handedness)
        ctl.update(
            gesture, cursor_point(fr.landmarks) if fr.landmarks is not None else None, fr.t,
            max(detector.strength.values()),
        )
        event = ctl.last_event if ctl.last_event != last_event else None
        last_event = ctl.last_event
        probs = detector.probs if fr.landmarks is not None else None
        top = max(probs, key=probs.get) if probs else None
        trace.append({
            "t": fr.t,
            "hand": fr.landmarks is not None,
            "gesture": gesture.value,
            "L": detector.strength[ctl.left_gesture],
            "R": detector.strength[ctl.right_gesture],
            "model": f"{top.value} {probs[top]:.0%}" if top else "-",
            "event": event[0] if event else "",
            "logged": fr.logged_event,
        })
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", nargs="?", type=Path)
    args = parser.parse_args()
    path = args.log
    if path is None:
        logs = sorted(config.DATA_DIR.glob("session-*.csv"))
        if not logs:
            raise SystemExit("No session logs. Record one: python -m gesture_control.main --log")
        path = logs[-1]

    frames, marks = load_log(path)
    classifier = GestureClassifier.load_if_available()
    trace = replay(frames, classifier)
    duration = frames[-1].t - frames[0].t if frames else 0.0
    print(f"{path.name}: {len(frames)} frames, {duration:.0f} s, "
          f"hand visible {np.mean([r['hand'] for r in trace]):.0%}, marks: {len(marks)}, "
          f"model: {'yes' if classifier else 'no'}")
    print("actions during recording:", dict(Counter(f.logged_event for f in frames if f.logged_event)))
    print("actions with current code:", dict(Counter(r["event"] for r in trace if r["event"])))

    for k, m in enumerate(marks, 1):
        print(f"\n=== MARK {k} at {m - frames[0].t:.1f} s - last {config.REPLAY_MARK_WINDOW_S:.0f} s ===")
        print(f"{'time':>6} {'gesture':13} {'L':>5} {'R':>5}  {'model':22} {'now':14} {'logged'}")
        for r in trace:
            if m - config.REPLAY_MARK_WINDOW_S <= r["t"] <= m:
                if r["event"] or r["logged"] or r["gesture"] != "point" or max(r["L"], r["R"]) > 0.3:
                    print(f"{r['t'] - m:6.2f} {r['gesture']:13} {r['L']:5.2f} {r['R']:5.2f}  "
                          f"{r['model']:22} {r['event']:14} {r['logged']}")


if __name__ == "__main__":
    main()
