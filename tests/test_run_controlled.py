from src.resource_controller import ActionType, ControlAction, ControllerState
from src.resource_monitor import ResourceSnapshot
from src.fault_injector import FaultScenario

from examples import run_controlled
from examples.run_controlled import (
    CSV_FIELDS,
    ControlRuntimeStats,
    apply_control_action,
    controlled_csv_row,
    latency_p95,
    should_skip_frame,
)


def make_snapshot() -> ResourceSnapshot:
    return ResourceSnapshot(
        timestamp=123.0,
        cpu_temp_celsius=41.25,
        cpu_usage_percent=12.5,
        cpu_freq_mhz=1800.0,
        memory_used_percent=33.3,
        memory_used_bytes=100,
        memory_available_bytes=200,
        is_throttled=False,
        throttle_flags=0,
        fps=29.75,
        pmic_rail_estimate_watts=1.23456,
    )


def make_action(
    action: ActionType,
    state: ControllerState = ControllerState.DEGRADED,
) -> ControlAction:
    return ControlAction(
        action=action,
        state=state,
        previous_state=ControllerState.NORMAL,
        reason="test reason",
        timestamp=123.0,
        source_snapshot=make_snapshot(),
    )


class FakeEstimator:
    def __init__(self, model: str = "thunder") -> None:
        self.model = model
        self.switches: list[str] = []

    def current_model(self) -> str:
        return self.model

    def switch_model(self, name: str) -> float:
        self.model = name
        self.switches.append(name)
        return 0.25


def test_latency_p95_uses_nearest_rank() -> None:
    assert latency_p95([]) is None
    assert latency_p95([10.0, 20.0, 30.0, 40.0]) == 40.0
    assert latency_p95([-1.0, 10.0]) == 10.0


def test_apply_control_action_switches_models() -> None:
    estimator = FakeEstimator("thunder")
    stats = ControlRuntimeStats()

    apply_control_action(make_action(ActionType.SWITCH_TO_LIGHT), estimator, stats)

    assert estimator.current_model() == "lightning"
    assert estimator.switches == ["lightning"]
    assert stats.model_switches == 1
    assert stats.last_switch_ms == 0.25

    apply_control_action(make_action(ActionType.SWITCH_TO_LIGHT), estimator, stats)

    assert estimator.switches == ["lightning"]
    assert stats.model_switches == 1
    assert stats.last_switch_ms == 0.0


def test_apply_control_action_force_gc(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(run_controlled.gc, "collect", lambda: calls.append("gc"))
    stats = ControlRuntimeStats()

    apply_control_action(make_action(ActionType.FORCE_GC), FakeEstimator(), stats)

    assert calls == ["gc"]
    assert stats.force_gc_count == 1


def test_should_skip_frame_only_in_critical() -> None:
    assert (
        should_skip_frame(
            state=ControllerState.DEGRADED,
            frame_sequence=2,
            frame_skip_ratio=0.5,
        )
        is False
    )
    assert (
        should_skip_frame(
            state=ControllerState.CRITICAL,
            frame_sequence=1,
            frame_skip_ratio=0.5,
        )
        is False
    )
    assert (
        should_skip_frame(
            state=ControllerState.CRITICAL,
            frame_sequence=2,
            frame_skip_ratio=0.5,
        )
        is True
    )


def test_controlled_csv_row_contains_control_columns() -> None:
    stats = ControlRuntimeStats(skipped_frames=2, model_switches=1, force_gc_count=1)
    stats.last_switch_ms = 0.25
    row = controlled_csv_row(
        snapshot=make_snapshot(),
        elapsed_s=2.3456,
        model_name="lightning",
        frames_processed=10,
        last_pose=None,
        control_action=make_action(ActionType.SWITCH_TO_LIGHT),
        stats=stats,
        recent_latencies_ms=[10.0, 20.0, 30.0, 40.0],
    )

    assert set(CSV_FIELDS) == set(row)
    assert row["model"] == "lightning"
    assert row["state"] == "degraded"
    assert row["previous_state"] == "normal"
    assert row["action"] == "switch_to_light"
    assert row["action_reason"] == "test reason"
    assert row["skipped_frames"] == 2
    assert row["model_switches"] == 1
    assert row["force_gc_count"] == 1
    assert row["switch_ms"] == 0.25
    assert row["recent_latency_p95_ms"] == 40.0
    assert row["fault_scenario"] == "none"
    assert row["fault_active"] is False


class FakeFaultInjector:
    def __init__(self) -> None:
        self.calls = []

    def inject_memory_pressure(self, **kwargs) -> None:
        self.calls.append(("memory", kwargs))

    def inject_cpu_stress(self, **kwargs) -> None:
        self.calls.append(("cpu", kwargs))


def test_start_configured_fault_starts_memory_pressure() -> None:
    args = type(
        "Args",
        (),
        {
            "fault_scenario": FaultScenario.MEMORY_PRESSURE.value,
            "fault_memory_target_percent": 85.0,
            "fault_duration": 3.0,
            "fault_cpu_workers": 2,
        },
    )()
    injector = FakeFaultInjector()

    run_controlled.start_configured_fault(args, injector)

    assert injector.calls == [
        (
            "memory",
            {
                "target_percent": 85.0,
                "duration_sec": 3.0,
            },
        )
    ]


def test_start_configured_fault_starts_cpu_stress() -> None:
    args = type(
        "Args",
        (),
        {
            "fault_scenario": FaultScenario.CPU_STRESS.value,
            "fault_memory_target_percent": 85.0,
            "fault_duration": 3.0,
            "fault_cpu_workers": 2,
        },
    )()
    injector = FakeFaultInjector()

    run_controlled.start_configured_fault(args, injector)

    assert injector.calls == [
        (
            "cpu",
            {
                "duration_sec": 3.0,
                "num_workers": 2,
            },
        )
    ]
