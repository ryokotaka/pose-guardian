import pytest

from src.resource_controller import (
    ActionType,
    ControllerState,
    Decision,
    ResourceController,
    choose_decision,
)
from src.resource_monitor import ResourceSnapshot


def make_snapshot(
    *,
    timestamp: float = 1.0,
    cpu_temp_celsius: float = 40.0,
    cpu_usage_percent: float = 10.0,
    memory_used_percent: float = 20.0,
    is_throttled: bool = False,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        timestamp=timestamp,
        cpu_temp_celsius=cpu_temp_celsius,
        cpu_usage_percent=cpu_usage_percent,
        cpu_freq_mhz=1800.0,
        memory_used_percent=memory_used_percent,
        memory_used_bytes=1_000_000,
        memory_available_bytes=3_000_000,
        is_throttled=is_throttled,
        throttle_flags=0x0E if is_throttled else 0,
        fps=30.0,
        pmic_rail_estimate_watts=None,
    )


def test_evaluate_stays_normal_under_normal_load() -> None:
    controller = ResourceController()

    action = controller.evaluate(make_snapshot())

    assert action.previous_state is ControllerState.NORMAL
    assert action.state is ControllerState.NORMAL
    assert action.action is ActionType.NONE
    assert action.reason == "within configured limits"


@pytest.mark.parametrize(
    ("snapshot", "reason"),
    [
        (
            make_snapshot(cpu_temp_celsius=72.0),
            "cpu_temp_celsius=72.0 >= threshold=70.0",
        ),
        (
            make_snapshot(memory_used_percent=85.0),
            "memory_used_percent=85.0 >= threshold=80.0",
        ),
    ],
)
def test_evaluate_degrades_on_temperature_or_memory(
    snapshot: ResourceSnapshot,
    reason: str,
) -> None:
    controller = ResourceController()

    action = controller.evaluate(snapshot)

    assert action.previous_state is ControllerState.NORMAL
    assert action.state is ControllerState.DEGRADED
    assert action.action is ActionType.SWITCH_TO_LIGHT
    assert reason in action.reason
    assert controller.state is ControllerState.DEGRADED


def test_evaluate_degrades_on_latency_slo_violation() -> None:
    controller = ResourceController()
    latencies = [210.0] * 20

    action = controller.evaluate(make_snapshot(), latencies)

    assert action.state is ControllerState.DEGRADED
    assert action.action is ActionType.SWITCH_TO_LIGHT
    assert "latency_p95_ms=210.0 >= threshold=200.0" in action.reason


def test_evaluate_ignores_latency_until_min_sample_count() -> None:
    controller = ResourceController()

    action = controller.evaluate(make_snapshot(), [500.0] * 19)

    assert action.state is ControllerState.NORMAL
    assert action.action is ActionType.NONE


def test_evaluate_escalates_to_critical_on_temperature() -> None:
    controller = ResourceController()

    action = controller.evaluate(make_snapshot(cpu_temp_celsius=83.4))

    assert action.previous_state is ControllerState.NORMAL
    assert action.state is ControllerState.CRITICAL
    assert action.action is ActionType.SKIP_FRAME
    assert "cpu_temp_celsius=83.4 >= threshold=80.0" in action.reason


def test_evaluate_escalates_to_critical_on_memory() -> None:
    controller = ResourceController()

    action = controller.evaluate(make_snapshot(memory_used_percent=95.0))

    assert action.state is ControllerState.CRITICAL
    assert action.action is ActionType.FORCE_GC
    assert "memory_used_percent=95.0 >= threshold=90.0" in action.reason


def test_evaluate_escalates_to_critical_when_currently_throttled() -> None:
    controller = ResourceController()

    action = controller.evaluate(make_snapshot(is_throttled=True))

    assert action.state is ControllerState.CRITICAL
    assert action.action is ActionType.SKIP_FRAME
    assert "is_throttled=True" in action.reason


def test_evaluate_escalates_from_degraded_to_critical() -> None:
    controller = ResourceController()
    degraded = controller.evaluate(make_snapshot(cpu_temp_celsius=72.0))

    critical = controller.evaluate(make_snapshot(cpu_temp_celsius=83.4))

    assert degraded.state is ControllerState.DEGRADED
    assert critical.previous_state is ControllerState.DEGRADED
    assert critical.state is ControllerState.CRITICAL
    assert critical.action is ActionType.SKIP_FRAME


def test_evaluate_does_not_recover_before_hysteresis_day() -> None:
    controller = ResourceController()
    degraded = controller.evaluate(make_snapshot(cpu_temp_celsius=72.0))

    normal_input = controller.evaluate(make_snapshot(cpu_temp_celsius=40.0))

    assert degraded.state is ControllerState.DEGRADED
    assert normal_input.previous_state is ControllerState.DEGRADED
    assert normal_input.state is ControllerState.DEGRADED
    assert normal_input.action is ActionType.NONE


def test_control_action_keeps_source_snapshot_and_timestamp() -> None:
    controller = ResourceController()
    snapshot = make_snapshot(timestamp=123.0, memory_used_percent=85.0)

    action = controller.evaluate(snapshot)

    assert action.timestamp == 123.0
    assert action.source_snapshot is snapshot


def test_legacy_choose_decision_uses_full_model_under_normal_load() -> None:
    decision = choose_decision(make_snapshot())

    assert decision is Decision.RUN_FULL


def test_legacy_choose_decision_falls_back_under_cpu_pressure() -> None:
    decision = choose_decision(make_snapshot(cpu_usage_percent=95.0))

    assert decision is Decision.RUN_TINY


def test_legacy_choose_decision_falls_back_under_memory_pressure() -> None:
    decision = choose_decision(make_snapshot(memory_used_percent=95.0))

    assert decision is Decision.RUN_TINY


def test_legacy_choose_decision_falls_back_when_currently_throttled() -> None:
    decision = choose_decision(make_snapshot(is_throttled=True))

    assert decision is Decision.RUN_TINY
