"""Pose estimator interface shared by Thunder and Lightning models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class ModelVariant(str, Enum):
    THUNDER = "thunder"
    LIGHTNING = "lightning"


@dataclass(frozen=True)
class Keypoint:
    name: str
    y: float
    x: float
    confidence: float


class PoseEstimator:
    """Thin placeholder for the Day 2 TFLite implementation."""

    def __init__(self, model_path: str | Path, variant: ModelVariant) -> None:
        self.model_path = Path(model_path)
        self.variant = variant

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return MoveNet-style keypoints with shape (17, 3)."""
        raise NotImplementedError("MoveNet inference is implemented on Day 2.")
