from pathlib import Path

from src.resource_monitor import ResourceSnapshot

from examples.run_monitored import CSV_FIELDS, csv_row, default_plot_path, parse_float


def test_default_plot_path_uses_metrics_plots_directory() -> None:
    path = default_plot_path(Path("metrics/monitored.csv"))

    assert str(path) == "metrics/plots/monitored.png"


def test_csv_row_contains_required_monitoring_columns() -> None:
    snapshot = ResourceSnapshot(
        timestamp=123.0,
        cpu_temp_celsius=41.25,
        cpu_usage_percent=12.5,
        cpu_freq_mhz=1800.0,
        memory_used_percent=33.3,
        memory_used_bytes=100,
        memory_available_bytes=200,
        is_throttled=True,
        throttle_flags=0xE,
        fps=29.75,
        pmic_rail_estimate_watts=1.23456,
    )

    row = csv_row(
        snapshot=snapshot,
        elapsed_s=2.3456,
        model_name="thunder",
        frames_processed=10,
        last_pose=None,
    )

    assert set(CSV_FIELDS) == set(row)
    assert row["cpu_temp"] == 41.25
    assert row["fps"] == 29.75
    assert row["throttle_flags"] == "0xe"
    assert row["pmic_rail_estimate_watts"] == 1.235


def test_parse_float_accepts_blank_and_numbers() -> None:
    assert parse_float("") is None
    assert parse_float("bad") is None
    assert parse_float("42.5") == 42.5
