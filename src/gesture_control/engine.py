"""Engine: camera -> hand tracking -> gestures -> mouse, as a reusable loop.

``Engine.run()`` blocks (used by ``python -m gesture_control.main``);
``Engine.start()`` runs it in a background thread for the tray app, which reads
``snapshot()`` for the indicator and sends commands (enable, pause, keyboard).
Commands are thread-safe; everything else lives on the engine thread.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, replace

import cv2

from gesture_control import config, keyboard
from gesture_control.camera import Camera, FpsCounter
from gesture_control.controller import ControlState, GestureController, HotkeyListener
from gesture_control.gestures import Gesture, GestureClassifier, GestureDetector
from gesture_control.mouse import (
    CursorAnimator, CursorMapper, Mouse, WorkZone, enable_dpi_awareness, screen_size,
)
from gesture_control.preview import draw_preview, recent_event, status_line
from gesture_control.tracker import HandTracker


def toggle_keyboard_async() -> None:
    """Opening osk.exe takes ~1 s; do it off the main loop so tracking keeps running."""
    threading.Thread(target=keyboard.toggle, name="keyboard", daemon=True).start()


@dataclass(frozen=True)
class Snapshot:
    """What the indicator / tray need to show, copied once per frame."""

    running: bool = False
    hand_visible: bool = False
    gesture: str = Gesture.NONE.value
    enabled: bool = True
    paused: bool = False
    keyboard_open: bool = False
    fps: float = 0.0
    event: str | None = None
    # normalized cursor point (for calibration), None without a hand
    cursor_point: tuple[float, float] | None = None
    model: bool = False
    error: str | None = None


class Engine:
    def __init__(self, log: bool = False) -> None:
        self.log = log
        self.state = ControlState()
        self._commands: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = Snapshot()
        self._thread: threading.Thread | None = None

    # -- control from other threads ------------------------------------------------
    def start(self) -> "Engine":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_safely, name="engine", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        self._thread = None

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def toggle_enabled(self) -> bool:
        return self.state.toggle()

    def toggle_pause(self) -> None:
        self._commands.put("pause")

    def toggle_keyboard(self) -> None:
        toggle_keyboard_async()

    def mark(self) -> None:
        self._commands.put("mark")

    # -- loop ------------------------------------------------------------------------
    def _publish(self, **changes) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, **changes)

    def _run_safely(self) -> None:
        try:
            self.run()
        except Exception as exc:  # shown by the tray app instead of a silent thread death
            self._publish(running=False, error=f"{type(exc).__name__}: {exc}")

    def run(self) -> None:
        """Blocking main loop; returns when stopped, the camera ends or q/Esc is pressed."""
        enable_dpi_awareness()
        width, height = screen_size()
        zone = WorkZone()
        mouse = Mouse()
        # Relative mode continues from wherever the real cursor is.
        mapper = CursorMapper(width, height, zone, position_source=lambda: mouse.position)
        animator = CursorAnimator(mouse)
        classifier = GestureClassifier.load_if_available()
        detector = GestureDetector(classifier)
        ctl = GestureController(mapper, animator, mouse, on_keyboard_toggle=toggle_keyboard_async)
        fps = FpsCounter()
        logger = None
        if self.log:
            from gesture_control.sessionlog import SessionLogger

            logger = SessionLogger()
            print(f"Logging landmarks and actions to {logger.path} (press m to mark a mistake)")
        if not keyboard.is_admin():
            print("Not running as administrator: on-screen keyboard clicks will be blocked by Windows.")
        print(f"Screen {width}x{height}. {config.HOTKEY_TOGGLE} toggles control.")
        print("Gestures:", "trained classifier + rules" if classifier else "rules only (no model yet)")

        was_enabled = True
        gesture = Gesture.NONE
        preview_open = False
        self._publish(running=True, model=classifier is not None, error=None)
        with Camera() as cam, HandTracker() as tracker, HotkeyListener(self.state), animator:
            try:
                while cam.is_running and not self._stop.is_set():
                    item = cam.read(timeout=1.0)
                    if item is None:
                        continue
                    frame, ts = item
                    hand = tracker.process(frame, ts)
                    while not self._commands.empty():
                        command = self._commands.get_nowait()
                        if command == "pause":
                            ctl.paused = not ctl.paused
                            ctl.release_all()
                            ctl.last_event = ("PAUSED" if ctl.paused else "RESUMED", ts)
                        elif command == "mark" and logger is not None:
                            logger.mark(ts)
                            ctl.last_event = ("MARKED", ts)
                    enabled = self.state.enabled

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
                    kb_open = keyboard.is_open()
                    self._publish(
                        hand_visible=hand is not None, gesture=gesture.value, enabled=enabled,
                        paused=ctl.paused, keyboard_open=kb_open, fps=current_fps,
                        event=recent_event(ctl, ts),
                        cursor_point=hand.cursor_point() if hand else None,
                    )

                    if config.SHOW_PREVIEW:
                        preview_open = True
                        cv2.imshow(
                            config.PREVIEW_WINDOW_NAME,
                            draw_preview(
                                frame, hand, zone, gesture, status_line(enabled, ctl), current_fps,
                                ctl.scroll_anchor,
                                max(detector.probs.values()) if detector.probs and hand else None,
                                detector.features,
                                (detector.strength, gesture, ctl.left_gesture, ctl.right_gesture),
                                recent_event(ctl, ts),
                                kb_open,
                            ),
                        )
                        key = cv2.waitKey(config.PREVIEW_WAIT_KEY_MS) & 0xFF
                        if key in config.PREVIEW_QUIT_KEYS:
                            break
                        if key == config.SESSION_MARK_KEY:
                            self.mark()
                    elif preview_open:
                        cv2.destroyWindow(config.PREVIEW_WINDOW_NAME)
                        preview_open = False
            finally:
                # Never leave a mouse button stuck down.
                ctl.release_all()
                if logger is not None:
                    logger.close()
                    print(f"Session log saved: {logger.path}")
                if preview_open:
                    cv2.destroyAllWindows()
                self._publish(running=False, hand_visible=False, cursor_point=None)
