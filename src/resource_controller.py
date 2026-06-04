"""Resource-aware inference control policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math

from src.resource_monitor import ResourceSnapshot


class ControllerState(str, Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class ActionType(str, Enum):
    NONE = "none"
    SWITCH_TO_LIGHT = "switch_to_light"
    SWITCH_TO_HEAVY = "switch_to_heavy"
    SKIP_FRAME = "skip_frame"
    REDUCE_FPS = "reduce_fps"
    RESTORE_FPS = "restore_fps"
    REDUCE_RESOLUTION = "reduce_resolution"
    FORCE_GC = "force_gc"


@dataclass(frozen=True)
class ControllerConfig:
    temp_degraded_celsius: float = 70.0
    temp_critical_celsius: float = 80.0
    temp_recover_normal_celsius: float = 60.0
    temp_recover_degraded_celsius: float = 65.0
    memory_degraded_percent: float = 80.0
    memory_critical_percent: float = 90.0
    memory_recover_normal_percent: float = 70.0
    memory_recover_degraded_percent: float = 75.0
    latency_slo_ms: float = 200.0
    latency_recover_ms: float = 140.0
    min_latency_samples: int = 20
    degraded_hold_sec: float = 10.0
    critical_hold_sec: float = 15.0
    frame_skip_ratio: float = 0.5


@dataclass(frozen=True)
class ControlAction:
    action: ActionType
    state: ControllerState
    previous_state: ControllerState
    reason: str
    timestamp: float
    source_snapshot: ResourceSnapshot


class ResourceController:
    """State machine that chooses the next resource-control action."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        if self.config.min_latency_samples <= 0:
            raise ValueError("min_latency_samples must be positive")
        self._state = ControllerState.NORMAL
        self._recovery_started_at: float | None = None
        self._recovery_target_state: ControllerState | None = None

    @property
    def state(self) -> ControllerState:
        return self._state

    def evaluate(
        self,
        snapshot: ResourceSnapshot,
        recent_latencies_ms: Sequence[float] | None = None,
    ) -> ControlAction:
        """Evaluate the snapshot and return the action for the next loop.

        Upward transitions happen immediately when pressure crosses configured
        limits. Recovery uses lower thresholds plus a hold timer so the
        controller does not chatter around a boundary.
        """
        previous_state = self._state
        pressure_state, reasons = self._pressure_state(snapshot, recent_latencies_ms)
        next_state, recovery_reason = self._next_state(
            previous_state,
            pressure_state,
            snapshot,
            recent_latencies_ms,
        )
        self._state = next_state
        action = self._action_for_transition(previous_state, next_state, reasons)
        reason = recovery_reason or (
            "; ".join(reasons) if reasons else "within configured limits"
        )
        return ControlAction(
            action=action,
            state=next_state,
            previous_state=previous_state,
            reason=reason,
            timestamp=snapshot.timestamp,
            source_snapshot=snapshot,
        )

    def _pressure_state(
        self,
        snapshot: ResourceSnapshot,
        recent_latencies_ms: Sequence[float] | None,
    ) -> tuple[ControllerState, list[str]]:
        latencies = tuple(recent_latencies_ms or ())
        critical_reasons: list[str] = []
        degraded_reasons: list[str] = []

        if snapshot.is_throttled:
            critical_reasons.append("is_throttled=True")
        if snapshot.cpu_temp_celsius >= self.config.temp_critical_celsius:
            critical_reasons.append(
                _threshold_reason(
                    "cpu_temp_celsius",
                    snapshot.cpu_temp_celsius,
                    self.config.temp_critical_celsius,
                )
            )
        elif snapshot.cpu_temp_celsius >= self.config.temp_degraded_celsius:
            degraded_reasons.append(
                _threshold_reason(
                    "cpu_temp_celsius",
                    snapshot.cpu_temp_celsius,
                    self.config.temp_degraded_celsius,
                )
            )

        if snapshot.memory_used_percent >= self.config.memory_critical_percent:
            critical_reasons.append(
                _threshold_reason(
                    "memory_used_percent",
                    snapshot.memory_used_percent,
                    self.config.memory_critical_percent,
                )
            )
        elif snapshot.memory_used_percent >= self.config.memory_degraded_percent:
            degraded_reasons.append(
                _threshold_reason(
                    "memory_used_percent",
                    snapshot.memory_used_percent,
                    self.config.memory_degraded_percent,
                )
            )

        p95 = _p95(latencies)
        if (
            p95 is not None
            and len(latencies) >= self.config.min_latency_samples
            and p95 > self.config.latency_slo_ms
        ):
            degraded_reasons.append(
                _threshold_reason("latency_p95_ms", p95, self.config.latency_slo_ms)
            )

        if critical_reasons:
            return ControllerState.CRITICAL, critical_reasons
        if degraded_reasons:
            return ControllerState.DEGRADED, degraded_reasons
        return ControllerState.NORMAL, []

    def _next_state(
        self,
        previous_state: ControllerState,
        pressure_state: ControllerState,
        snapshot: ResourceSnapshot,
        recent_latencies_ms: Sequence[float] | None,
    ) -> tuple[ControllerState, str | None]:
        if pressure_state is ControllerState.CRITICAL:
            self._clear_recovery_timer()
            return ControllerState.CRITICAL, None
        if pressure_state is ControllerState.DEGRADED:
            if previous_state is ControllerState.CRITICAL:
                return self._recover_with_hold(
                    target_state=ControllerState.DEGRADED,
                    required_hold_sec=self.config.critical_hold_sec,
                    conditions=self._can_recover_to_degraded(snapshot),
                    timestamp=snapshot.timestamp,
                )
            self._clear_recovery_timer()
            return ControllerState.DEGRADED, None

        if previous_state is ControllerState.CRITICAL:
            return self._recover_with_hold(
                target_state=ControllerState.DEGRADED,
                required_hold_sec=self.config.critical_hold_sec,
                conditions=self._can_recover_to_degraded(snapshot),
                timestamp=snapshot.timestamp,
            )
        if previous_state is ControllerState.DEGRADED:
            return self._recover_with_hold(
                target_state=ControllerState.NORMAL,
                required_hold_sec=self.config.degraded_hold_sec,
                conditions=self._can_recover_to_normal(snapshot, recent_latencies_ms),
                timestamp=snapshot.timestamp,
            )

        self._clear_recovery_timer()
        return previous_state, None

    @staticmethod
    def _action_for_transition(
        previous_state: ControllerState,
        next_state: ControllerState,
        reasons: Sequence[str],
    ) -> ActionType:
        if previous_state is next_state:
            return ActionType.NONE
        if next_state is ControllerState.DEGRADED:
            return ActionType.SWITCH_TO_LIGHT
        if next_state is ControllerState.CRITICAL:
            if any(reason.startswith("memory_used_percent=") for reason in reasons):
                return ActionType.FORCE_GC
            return ActionType.SKIP_FRAME
        if next_state is ControllerState.NORMAL:
            return ActionType.SWITCH_TO_HEAVY
        return ActionType.NONE

    def _recover_with_hold(
        self,
        *,
        target_state: ControllerState,
        required_hold_sec: float,
        conditions: list[str],
        timestamp: float,
    ) -> tuple[ControllerState, str | None]:
        if not conditions:
            self._clear_recovery_timer()
            return self._state, "recovery blocked: recovery thresholds not met"

        if self._recovery_target_state is not target_state:
            self._recovery_target_state = target_state
            self._recovery_started_at = timestamp

        assert self._recovery_started_at is not None
        held_sec = max(0.0, timestamp - self._recovery_started_at)
        reason = (
            f"recovery to {target_state.value}: "
            f"{'; '.join(conditions)}; "
            f"held={held_sec:.1f}s required={required_hold_sec:.1f}s"
        )
        if held_sec >= required_hold_sec:
            self._clear_recovery_timer()
            return target_state, reason
        return self._state, reason

    def _clear_recovery_timer(self) -> None:
        self._recovery_started_at = None
        self._recovery_target_state = None

    def _can_recover_to_normal(
        self,
        snapshot: ResourceSnapshot,
        recent_latencies_ms: Sequence[float] | None,
    ) -> list[str]:
        conditions: list[str] = []
        if snapshot.is_throttled:
            return []
        if snapshot.cpu_temp_celsius >= self.config.temp_recover_normal_celsius:
            return []
        if snapshot.memory_used_percent >= self.config.memory_recover_normal_percent:
            return []

        latencies = tuple(recent_latencies_ms or ())
        p95 = _p95(latencies)
        if len(latencies) >= self.config.min_latency_samples:
            if p95 is None or p95 >= self.config.latency_recover_ms:
                return []
            conditions.append(
                _recovery_reason(
                    "latency_p95_ms",
                    p95,
                    self.config.latency_recover_ms,
                )
            )

        conditions.extend(
            [
                _recovery_reason(
                    "cpu_temp_celsius",
                    snapshot.cpu_temp_celsius,
                    self.config.temp_recover_normal_celsius,
                ),
                _recovery_reason(
                    "memory_used_percent",
                    snapshot.memory_used_percent,
                    self.config.memory_recover_normal_percent,
                ),
            ]
        )
        return conditions

    def _can_recover_to_degraded(self, snapshot: ResourceSnapshot) -> list[str]:
        if snapshot.is_throttled:
            return []
        if snapshot.cpu_temp_celsius >= self.config.temp_recover_degraded_celsius:
            return []
        if snapshot.memory_used_percent >= self.config.memory_recover_degraded_percent:
            return []
        return [
            _recovery_reason(
                "cpu_temp_celsius",
                snapshot.cpu_temp_celsius,
                self.config.temp_recover_degraded_celsius,
            ),
            _recovery_reason(
                "memory_used_percent",
                snapshot.memory_used_percent,
                self.config.memory_recover_degraded_percent,
            ),
        ]


def _threshold_reason(metric: str, value: float, threshold: float) -> str:
    return f"{metric}={value:.1f} >= threshold={threshold:.1f}"


def _recovery_reason(metric: str, value: float, threshold: float) -> str:
    return f"{metric}={value:.1f} < recovery={threshold:.1f}"


def _p95(values: Sequence[float]) -> float | None:
    clean_values = sorted(float(value) for value in values if value >= 0)
    if not clean_values:
        return None
    index = math.ceil(0.95 * len(clean_values)) - 1
    return clean_values[max(0, min(index, len(clean_values) - 1))]


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


def choose_decision(
    snapshot: ResourceSnapshot,
    limits: ControllerLimits | None = None,
) -> Decision:
    """Legacy two-way policy kept for older callers."""
    limits = limits or ControllerLimits()
    if snapshot.is_throttled:
        return Decision.RUN_TINY
    if (
        snapshot.cpu_usage_percent >= limits.cpu_high_percent
        or snapshot.memory_used_percent >= limits.memory_high_percent
    ):
        return Decision.RUN_TINY
    return Decision.RUN_FULL
