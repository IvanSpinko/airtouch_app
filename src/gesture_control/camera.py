"""Webcam frame capture (OpenCV).

Frames are read in a background thread and only the latest frame is handed out,
so processing never builds up a queue and latency does not grow.

Check: ``python -m gesture_control.camera`` shows a window with video and FPS.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np

from gesture_control import config


class CameraError(RuntimeError):
    pass


class Camera:
    """Threaded frame capture. Use as a context manager."""

    def __init__(
        self,
        index: int | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        mirror: bool | None = None,
    ) -> None:
        self.index = config.CAMERA_INDEX if index is None else index
        self.width = config.CAMERA_WIDTH if width is None else width
        self.height = config.CAMERA_HEIGHT if height is None else height
        self.fps = config.CAMERA_FPS if fps is None else fps
        self.mirror = config.CAMERA_MIRROR if mirror is None else mirror

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._running = False

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> "Camera":
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            raise CameraError(f"Cannot open camera #{self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._reader, name="camera", daemon=True)
        self._thread.start()
        if not self._new_frame.wait(config.CAMERA_FIRST_FRAME_TIMEOUT_S):
            self.stop()
            raise CameraError("Camera opened but delivers no frames")
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- reading -----------------------------------------------------------------
    def _reader(self) -> None:
        failed = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                failed += 1
                if failed >= config.CAMERA_MAX_FAILED_READS:
                    self._running = False
                continue
            failed = 0
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
                self._frame_time = time.perf_counter()
            self._new_frame.set()

    @property
    def is_running(self) -> bool:
        return self._running

    def read(self, timeout: float | None = None) -> tuple[np.ndarray, float] | None:
        """Waits for a new frame and returns (BGR frame, capture time from perf_counter).

        Returns None if the camera stopped or the timeout expired.
        """
        if not self._new_frame.wait(timeout):
            return None
        with self._lock:
            self._new_frame.clear()
            if self._frame is None or not self._running:
                return None
            return self._frame, self._frame_time


class FpsCounter:
    """Smoothed frames-per-second counter."""

    def __init__(self, smoothing: float = config.FPS_SMOOTHING) -> None:
        self.smoothing = smoothing
        self.fps = 0.0
        self._last: float | None = None

    def tick(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        if self._last is not None and now > self._last:
            inst = 1.0 / (now - self._last)
            self.fps = inst if self.fps == 0.0 else (
                self.smoothing * self.fps + (1.0 - self.smoothing) * inst
            )
        self._last = now
        return self.fps


def _demo() -> None:
    fps = FpsCounter()
    with Camera() as cam:
        while cam.is_running:
            item = cam.read(timeout=1.0)
            if item is None:
                continue
            frame, _ = item
            frame = frame.copy()
            cv2.putText(
                frame, f"FPS: {fps.tick():.1f}", config.PREVIEW_TEXT_ORIGIN,
                cv2.FONT_HERSHEY_SIMPLEX, config.PREVIEW_FONT_SCALE,
                config.COLOR_TEXT_ACTIVE, config.PREVIEW_FONT_THICKNESS,
            )
            cv2.imshow("camera", frame)
            if cv2.waitKey(config.PREVIEW_WAIT_KEY_MS) & 0xFF in config.PREVIEW_QUIT_KEYS:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
