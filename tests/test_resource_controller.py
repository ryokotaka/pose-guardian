from src.resource_controller import Decision, choose_decision
from src.resource_monitor import ResourceSnapshot


def test_controller_uses_full_model_under_normal_load() -> None:
    decision = choose_decision(ResourceSnapshot(cpu_percent=10.0, memory_percent=20.0))

    assert decision is Decision.RUN_FULL


def test_controller_falls_back_under_resource_pressure() -> None:
    decision = choose_decision(ResourceSnapshot(cpu_percent=95.0, memory_percent=20.0))

    assert decision is Decision.RUN_TINY
