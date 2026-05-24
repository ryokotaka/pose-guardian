"""Camera input helpers."""

from __future__ import annotations

import cv2


def open_camera(device_index: int = 0) -> cv2.VideoCapture:
    """Open a local camera device and fail fast when it is unavailable."""
    capture = cv2.VideoCapture(device_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera device {device_index}")
    return capture
