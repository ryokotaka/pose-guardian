"""Fault injection hooks for later benchmark scenarios."""

from __future__ import annotations

from enum import Enum


class FaultScenario(str, Enum):
    NONE = "none"
    CAMERA_DISCONNECT = "camera_disconnect"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_STRESS = "cpu_stress"


def should_inject(frame_index: int, every_n_frames: int | None) -> bool:
    if every_n_frames is None or every_n_frames <= 0:
        return False
    return frame_index > 0 and frame_index % every_n_frames == 0
