"""Configuration defaults for local development."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    heavy_model_path: Path = Path("models/movenet_thunder.tflite")
    light_model_path: Path = Path("models/movenet_lightning.tflite")
    camera_device_index: int = 0
    target_fps: int = 30
