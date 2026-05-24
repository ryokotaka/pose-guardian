"""Resource monitoring helpers for CPU and memory pressure."""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_percent: float
    memory_percent: float


def read_resources() -> ResourceSnapshot:
    return ResourceSnapshot(
        cpu_percent=psutil.cpu_percent(interval=None),
        memory_percent=psutil.virtual_memory().percent,
    )
