"""Drawing utilities for pose keypoints."""

from __future__ import annotations

import cv2
import numpy as np


def draw_keypoints(image_bgr: np.ndarray, keypoints: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    output = image_bgr.copy()
    height, width = output.shape[:2]
    for y, x, confidence in keypoints:
        if confidence < threshold:
            continue
        center = (int(x * width), int(y * height))
        cv2.circle(output, center, 4, (0, 255, 0), -1)
    return output
