import time

import pytest

from src import resource_monitor
from src.resource_monitor import (
    MonitorConfig,
    ResourceMonitor,
    ResourceSnapshot,
    _normalize_pmic_rail_name,
    _parse_get_throttled,
    _parse_measure_temp,
    _parse_pmic_read_adc,
    is_currently_throttled,
    read_resources,
)


def test_parse_measure_temp() -> None:
    assert _parse_measure_temp("temp=42.3'C") == 42.3
    assert _parse_measure_temp("bad output") is None


def test_parse_get_throttled_hex_and_decimal() -> None:
    assert _parse_get_throttled("throttled=0x50005") == 0x50005
    assert _parse_get_throttled("throttled=8") == 8
    assert _parse_get_throttled("bad output") == 0


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (0x0, False),
        (0x1, False),  # current undervoltage is tracked but not thermal throttle.
        (0x2, True),
        (0x4, True),
        (0x8, True),
        (0x50000, False),  # historical bits only.
        (0xE0008, True),
    ],
)
def test_is_currently_throttled(flags: int, expected: bool) -> None:
    assert is_currently_throttled(flags) is expected


def test_parse_pmic_read_adc_estimates_matching_rails() -> None:
    output = """
       3V3_SYS_A current(1)=0.50000000A
       VDD_CORE_A current(7)=1.00000000A
       3V3_SYS_V volt(9)=3.30000000V
       VDD_CORE_V volt(15)=0.85000000V
    """

    assert _parse_pmic_read_adc(output) == pytest.approx(2.5)


def test_parse_pmic_read_adc_returns_none_without_pairs() -> None:
    assert _parse_pmic_read_adc("EXT5V_V volt(24)=5.08664000V") is None


def test_normalize_pmic_rail_name() -> None:
    assert _normalize_pmic_rail_name("3V3_SYS_A") == "3V3_SYS"
    assert _normalize_pmic_rail_name("3V3_SYS_V") == "3V3_SYS"
    assert _normalize_pmic_rail_name("BATT") == "BATT"


def test_read_resources_returns_full_snapshot() -> None:
    snapshot = read_resources()

    assert isinstance(snapshot, ResourceSnapshot)
    assert snapshot.timestamp > 0
    assert snapshot.cpu_temp_celsius >= 0.0
    assert snapshot.cpu_usage_percent >= 0.0
    assert snapshot.memory_used_percent >= 0.0
    assert snapshot.memory_used_bytes >= 0
    assert snapshot.memory_available_bytes >= 0
    assert snapshot.fps == 0.0


def test_monitor_record_inference_updates_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([10.0, 10.0, 10.5, 11.0])
    monkeypatch.setattr(resource_monitor.time, "monotonic", lambda: next(times))

    monitor = ResourceMonitor(MonitorConfig(enable_power=False))
    monitor.record_inference(30.0)
    monitor.record_inference(30.0)
    snapshot = monitor._sample()

    assert snapshot.fps == pytest.approx(2.0)


def test_monitor_history_filters_by_age(monkeypatch: pytest.MonkeyPatch) -> None:
    current_time = 100.0
    monkeypatch.setattr(resource_monitor.time, "monotonic", lambda: current_time)
    monitor = ResourceMonitor(MonitorConfig(enable_power=False))

    old = ResourceSnapshot(
        timestamp=90.0,
        cpu_temp_celsius=40.0,
        cpu_usage_percent=1.0,
        cpu_freq_mhz=1000.0,
        memory_used_percent=2.0,
        memory_used_bytes=1,
        memory_available_bytes=2,
        is_throttled=False,
        throttle_flags=0,
        fps=0.0,
        pmic_rail_estimate_watts=None,
    )
    new = ResourceSnapshot(
        timestamp=99.0,
        cpu_temp_celsius=41.0,
        cpu_usage_percent=1.0,
        cpu_freq_mhz=1000.0,
        memory_used_percent=2.0,
        memory_used_bytes=1,
        memory_available_bytes=2,
        is_throttled=False,
        throttle_flags=0,
        fps=0.0,
        pmic_rail_estimate_watts=None,
    )
    monitor._store_snapshot(old)
    monitor._store_snapshot(new)

    history = monitor.history(seconds=5)
    assert old not in history
    assert new in history


def test_monitor_start_stop_is_idempotent() -> None:
    monitor = ResourceMonitor(MonitorConfig(sample_interval_sec=0.01, enable_power=False))

    monitor.start()
    monitor.start()
    time.sleep(0.03)
    monitor.stop()
    monitor.stop()

    assert monitor.snapshot().timestamp > 0
