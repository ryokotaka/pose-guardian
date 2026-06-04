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


def test_evaluate_recovers_from_degraded_after_hold_time() -> None:
    controller = ResourceController()
    degraded = controller.evaluate(make_snapshot(timestamp=0.0, cpu_temp_celsius=72.0))

    pending = controller.evaluate(make_snapshot(timestamp=1.0, cpu_temp_celsius=55.0))
    recovered = controller.evaluate(make_snapshot(timestamp=11.0, cpu_temp_celsius=55.0))

    assert degraded.state is ControllerState.DEGRADED
    assert pending.previous_state is ControllerState.DEGRADED
    assert pending.state is ControllerState.DEGRADED
    assert pending.action is ActionType.NONE
    assert "held=0.0s required=10.0s" in pending.reason
    assert recovered.previous_state is ControllerState.DEGRADED
    assert recovered.state is ControllerState.NORMAL
    assert recovered.action is ActionType.SWITCH_TO_HEAVY
    assert "held=10.0s required=10.0s" in recovered.reason


def test_recovery_timer_resets_when_recovery_condition_breaks() -> None:
    controller = ResourceController()
    controller.evaluate(make_snapshot(timestamp=0.0, cpu_temp_celsius=72.0))
    controller.evaluate(make_snapshot(timestamp=1.0, cpu_temp_celsius=55.0))

    blocked = controller.evaluate(make_snapshot(timestamp=5.0, cpu_temp_celsius=61.0))
    restarted = controller.evaluate(make_snapshot(timestamp=6.0, cpu_temp_celsius=55.0))
    still_pending = controller.evaluate(make_snapshot(timestamp=15.0, cpu_temp_celsius=55.0))
    recovered = controller.evaluate(make_snapshot(timestamp=16.0, cpu_temp_celsius=55.0))

    assert blocked.state is ControllerState.DEGRADED
    assert "recovery blocked" in blocked.reason
    assert restarted.state is ControllerState.DEGRADED
    assert "held=0.0s required=10.0s" in restarted.reason
    assert still_pending.state is ControllerState.DEGRADED
    assert "held=9.0s required=10.0s" in still_pending.reason
    assert recovered.state is ControllerState.NORMAL


def test_temperature_chatter_near_degraded_threshold_does_not_recover() -> None:
    controller = ResourceController()
    controller.evaluate(make_snapshot(timestamp=0.0, cpu_temp_celsius=72.0))

    for timestamp, temp in [(1.0, 69.0), (2.0, 71.0), (3.0, 59.0), (4.0, 61.0)]:
        action = controller.evaluate(
            make_snapshot(timestamp=timestamp, cpu_temp_celsius=temp)
        )

        assert action.state is ControllerState.DEGRADED
        assert action.action is ActionType.NONE


def test_evaluate_recovers_from_critical_to_degraded_after_hold_time() -> None:
    controller = ResourceController()
    critical = controller.evaluate(make_snapshot(timestamp=0.0, cpu_temp_celsius=83.4))

    pending = controller.evaluate(make_snapshot(timestamp=1.0, cpu_temp_celsius=60.0))
    recovered = controller.evaluate(make_snapshot(timestamp=16.0, cpu_temp_celsius=60.0))

    assert critical.state is ControllerState.CRITICAL
    assert pending.state is ControllerState.CRITICAL
    assert "held=0.0s required=15.0s" in pending.reason
    assert recovered.previous_state is ControllerState.CRITICAL
    assert recovered.state is ControllerState.DEGRADED
    assert recovered.action is ActionType.SWITCH_TO_LIGHT
    assert "held=15.0s required=15.0s" in recovered.reason


def test_latency_must_recover_below_recovery_threshold() -> None:
    controller = ResourceController()
    slow = [210.0] * 20
    medium = [150.0] * 20
    fast = [100.0] * 20
    controller.evaluate(make_snapshot(timestamp=0.0), slow)

    blocked = controller.evaluate(make_snapshot(timestamp=20.0), medium)
    pending = controller.evaluate(make_snapshot(timestamp=21.0), fast)
    recovered = controller.evaluate(make_snapshot(timestamp=31.0), fast)

    assert blocked.state is ControllerState.DEGRADED
    assert "recovery blocked" in blocked.reason
    assert pending.state is ControllerState.DEGRADED
    assert "latency_p95_ms=100.0 < recovery=140.0" in pending.reason
    assert recovered.state is ControllerState.NORMAL
    assert recovered.action is ActionType.SWITCH_TO_HEAVY


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
