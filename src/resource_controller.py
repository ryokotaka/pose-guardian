"""Resource-aware inference decision policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.resource_monitor import ResourceSnapshot


class Decision(str, Enum):
    RUN_FULL = "run_full"
    RUN_TINY = "run_tiny"
    QUEUE = "queue"
    SKIP = "skip"
    SAFE_MODE = "safe_mode"
    REJECT = "reject"


@dataclass(frozen=True)
class ControllerLimits:
    cpu_high_percent: float = 85.0
    memory_high_percent: float = 85.0


def choose_decision(snapshot: ResourceSnapshot, limits: ControllerLimits | None = None) -> Decision:
    limits = limits or ControllerLimits()
    if snapshot.cpu_percent >= limits.cpu_high_percent or snapshot.memory_percent >= limits.memory_high_percent:
        return Decision.RUN_TINY
    return Decision.RUN_FULL
