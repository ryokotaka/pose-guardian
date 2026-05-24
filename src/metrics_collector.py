"""Small in-memory metrics collector used before CSV/JSON export lands."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class MetricsCollector:
    started_at: float = field(default_factory=perf_counter)
    frames_seen: int = 0

    def record_frame(self) -> None:
        self.frames_seen += 1

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self.started_at
