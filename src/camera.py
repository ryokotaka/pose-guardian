"""Threaded camera capture (always exposes the latest frame).

Spec: 00_機能仕様書.md 3.1 Camera

The capture thread reads from OpenCV continuously and keeps only the
**latest** frame so the inference loop never reads a stale buffered frame.
Only the ``opencv`` source is implemented in this baseline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for :class:`Camera`."""

    source: str = "opencv"        # baseline supports "opencv"
    device_index: int = 0          # 0: built-in or first USB camera
    width: int = 640
    height: int = 480
    fps_cap: int = 30


class Camera:
    """OpenCV-backed camera with a background capture thread.

    Use as a context manager::

        with Camera(CameraConfig()) as cam:
            frame = cam.read_frame()
    """

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames_read = 0
        self._frames_dropped = 0
        # 単調増加するフレーム ID。 ``read_new_frame`` がこれを使って
        # 「同じフレームの再処理」を呼び出し側が検出できるようにする。
        self._frame_id = 0

    # ---- lifecycle ----
    def start(self) -> None:
        if self.config.source != "opencv":
            raise NotImplementedError(
                f"Unsupported camera source: {self.config.source}. "
                "This baseline implements 'opencv' only."
            )
        if self._thread is not None:
            return  # already started, idempotent

        cap = cv2.VideoCapture(self.config.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps_cap)
        # macOS の VideoCapture では効かないことが多いが念のため
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Unable to open camera device {self.config.device_index}. "
                "On macOS, grant camera permission to the running process "
                "(System Settings -> Privacy & Security -> Camera) and "
                "restart the terminal."
            )

        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="CameraThread",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ---- internal ----
    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._frames_dropped += 1
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = frame  # BGR uint8
                self._frames_read += 1
                self._frame_id += 1

    # ---- public read API ----
    def read_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent frame, or ``None`` if none yet.

        Note: This returns the same frame on consecutive calls if no new
        frame has arrived. Prefer :meth:`read_new_frame` for inference
        loops to avoid re-processing the same frame.
        """
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def read_new_frame(self, last_id: int) -> Tuple[Optional[np.ndarray], int]:
        """Return ``(frame, frame_id)`` only if a new frame has arrived.

        ``last_id`` is the frame ID the caller last processed (initially 0).
        If the latest frame is the same one or no frame yet, returns
        ``(None, last_id)``. Otherwise returns ``(frame_copy, current_id)``.

        Use in inference loops::

            last_id = 0
            while True:
                frame, current_id = cam.read_new_frame(last_id)
                if frame is None:
                    time.sleep(0.005); continue
                last_id = current_id
                ...
        """
        with self._lock:
            if self._latest is None or self._frame_id == last_id:
                return None, last_id
            return self._latest.copy(), self._frame_id

    def is_alive(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def stats(self) -> dict[str, object]:
        return {
            "frames_read": self._frames_read,
            "frames_dropped": self._frames_dropped,
            "frame_id": self._frame_id,
            "is_alive": self.is_alive(),
        }

    # ---- context manager ----
    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def open_camera(device_index: int = 0) -> cv2.VideoCapture:
    """Small compatibility helper for direct OpenCV capture.

    Prefer :class:`Camera` for any new code that needs realtime capture.
    """
    capture = cv2.VideoCapture(device_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open camera device {device_index}")
    return capture
