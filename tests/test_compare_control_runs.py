from pathlib import Path

from examples.compare_control_runs import (
    markdown_table,
    parse_bool,
    parse_float,
    summarize_run,
)


def test_parse_helpers() -> None:
    assert parse_float("1.25") == 1.25
    assert parse_float("") is None
    assert parse_float("not-a-number") is None
    assert parse_bool(True) is True
    assert parse_bool("true") is True
    assert parse_bool("False") is False


def test_summarize_run_counts_slo_and_modes() -> None:
    rows = [
        {
            "elapsed_s": "0.0",
            "controller_mode": "naive",
            "model": "thunder",
            "state": "normal",
            "action": "none",
            "fault_active": "False",
            "recent_latency_p95_ms": "100.0",
            "inference_ms": "40.0",
            "fps": "24.0",
            "cpu_temp": "50.0",
            "is_throttled": "False",
            "model_switches": "0",
        },
        {
            "elapsed_s": "1.0",
            "controller_mode": "naive",
            "model": "thunder",
            "state": "normal",
            "action": "none",
            "fault_active": "True",
            "recent_latency_p95_ms": "220.0",
            "inference_ms": "80.0",
            "fps": "12.0",
            "cpu_temp": "55.0",
            "is_throttled": "False",
            "model_switches": "0",
        },
    ]

    summary = summarize_run(
        label="naive",
        csv_path=Path("metrics/naive.csv"),
        rows=rows,
        slo_ms=200.0,
    )

    assert summary.rows == 2
    assert summary.duration_s == 1.0
    assert summary.controller_modes == {"naive": 2}
    assert summary.models == {"thunder": 2}
    assert summary.fault_active_rows == 1
    assert summary.slo_violation_rows == 1
    assert summary.slo_violation_pct == 50.0
    assert summary.recent_latency_p95_avg_ms == 160.0
    assert summary.recent_latency_p95_max_ms == 220.0
    assert summary.inference_avg_ms == 60.0
    assert summary.fps_min == 12.0
    assert summary.temp_max_c == 55.0
    assert summary.throttle_rows == 0
    assert summary.final_model_switches == 0


def test_markdown_table_contains_comparison_columns() -> None:
    summary = summarize_run(
        label="controlled",
        csv_path=Path("metrics/controlled.csv"),
        rows=[
            {
                "elapsed_s": "1.0",
                "controller_mode": "controlled",
                "model": "lightning",
                "state": "degraded",
                "action": "switch_to_light",
                "fault_active": "True",
                "recent_latency_p95_ms": "201.0",
                "inference_ms": "12.0",
                "fps": "24.0",
                "cpu_temp": "60.0",
                "is_throttled": "False",
                "model_switches": "1",
            }
        ],
        slo_ms=200.0,
    )

    table = markdown_table([summary])

    assert "SLO_rows" in table
    assert "controlled" in table
    assert "switch_to_light:1" in table
    assert "lightning:1" in table
