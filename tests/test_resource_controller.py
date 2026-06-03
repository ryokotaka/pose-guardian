from src.resource_controller import Decision, choose_decision
from src.resource_monitor import ResourceSnapshot


def make_snapshot(
    *,
    cpu_usage_percent: float = 10.0,
    memory_used_percent: float = 20.0,
    is_throttled: bool = False,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        timestamp=1.0,
        cpu_temp_celsius=40.0,
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


def test_controller_uses_full_model_under_normal_load() -> None:
    decision = choose_decision(make_snapshot())

    assert decision is Decision.RUN_FULL


def test_controller_falls_back_under_cpu_pressure() -> None:
    decision = choose_decision(make_snapshot(cpu_usage_percent=95.0))

    assert decision is Decision.RUN_TINY


def test_controller_falls_back_under_memory_pressure() -> None:
    decision = choose_decision(make_snapshot(memory_used_percent=95.0))

    assert decision is Decision.RUN_TINY


def test_controller_falls_back_when_currently_throttled() -> None:
    decision = choose_decision(make_snapshot(is_throttled=True))

    assert decision is Decision.RUN_TINY
