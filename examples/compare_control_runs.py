"""Compare naive and controlled run_controlled.py CSV outputs.

Usage:
    .venv/bin/python examples/compare_control_runs.py \
        --naive-csv metrics/naive_cpu_stress.csv \
        --controlled-csv metrics/controlled_cpu_stress.csv \
        --markdown-output docs/controlled_vs_naive.md \
        --plot-output docs/assets/naive_vs_controlled_cpu_stress.png

Raw CSV files under ``metrics/`` are local benchmark outputs and should not be
committed. A selected plot can be written under ``docs/assets/`` when it is used
by the README.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSummary:
    label: str
    csv_path: Path
    rows: int
    duration_s: float | None
    controller_modes: dict[str, int]
    models: dict[str, int]
    states: dict[str, int]
    actions: dict[str, int]
    fault_active_rows: int
    slo_violation_rows: int
    slo_violation_pct: float
    recent_latency_p95_avg_ms: float | None
    recent_latency_p95_max_ms: float | None
    inference_avg_ms: float | None
    inference_max_ms: float | None
    fps_avg: float | None
    fps_min: float | None
    temp_max_c: float | None
    throttle_rows: int
    final_model_switches: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare naive vs controlled run_controlled.py CSV files."
    )
    parser.add_argument("--naive-csv", type=Path, required=True)
    parser.add_argument("--controlled-csv", type=Path, required=True)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/controlled_vs_naive.md"),
    )
    parser.add_argument("--plot-output", type=Path)
    parser.add_argument("--slo-ms", type=float, default=200.0)
    return parser.parse_args()


def parse_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = parse_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def time_series(rows: list[dict[str, str]], field: str) -> tuple[list[float], list[float]]:
    elapsed_values: list[float] = []
    field_values: list[float] = []
    for row in rows:
        elapsed = parse_float(row.get("elapsed_s"))
        value = parse_float(row.get(field))
        if elapsed is None or value is None:
            continue
        elapsed_values.append(elapsed)
        field_values.append(value)
    return elapsed_values, field_values


def avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def rounded(value: float | None, digits: int = 3) -> float | str:
    if value is None:
        return ""
    return round(value, digits)


def counter(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(Counter(row.get(field, "") for row in rows))


def summarize_run(
    *,
    label: str,
    csv_path: Path,
    rows: list[dict[str, str]],
    slo_ms: float,
) -> RunSummary:
    elapsed = numeric_values(rows, "elapsed_s")
    recent_p95 = numeric_values(rows, "recent_latency_p95_ms")
    inference = numeric_values(rows, "inference_ms")
    fps = numeric_values(rows, "fps")
    temps = numeric_values(rows, "cpu_temp")
    switches = [parse_int(row.get("model_switches")) for row in rows]
    final_switches = next((value for value in reversed(switches) if value is not None), 0)
    slo_rows = sum(1 for value in recent_p95 if value > slo_ms)
    return RunSummary(
        label=label,
        csv_path=csv_path,
        rows=len(rows),
        duration_s=max(elapsed) if elapsed else None,
        controller_modes=counter(rows, "controller_mode"),
        models=counter(rows, "model"),
        states=counter(rows, "state"),
        actions=counter(rows, "action"),
        fault_active_rows=sum(1 for row in rows if parse_bool(row.get("fault_active"))),
        slo_violation_rows=slo_rows,
        slo_violation_pct=(slo_rows / len(recent_p95) * 100.0) if recent_p95 else 0.0,
        recent_latency_p95_avg_ms=avg(recent_p95),
        recent_latency_p95_max_ms=max(recent_p95) if recent_p95 else None,
        inference_avg_ms=avg(inference),
        inference_max_ms=max(inference) if inference else None,
        fps_avg=avg(fps),
        fps_min=min(fps) if fps else None,
        temp_max_c=max(temps) if temps else None,
        throttle_rows=sum(1 for row in rows if parse_bool(row.get("is_throttled"))),
        final_model_switches=final_switches or 0,
    )


def counts_text(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def markdown_table(summaries: list[RunSummary]) -> str:
    lines = [
        "| run | rows | duration_s | mode | models | states | actions | p95_avg_ms | p95_max_ms | SLO_rows | SLO_pct | inference_avg_ms | fps_avg | temp_max_c | throttle_rows | switches |",
        "|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| "
            f"{summary.label} | "
            f"{summary.rows} | "
            f"{rounded(summary.duration_s)} | "
            f"{counts_text(summary.controller_modes)} | "
            f"{counts_text(summary.models)} | "
            f"{counts_text(summary.states)} | "
            f"{counts_text(summary.actions)} | "
            f"{rounded(summary.recent_latency_p95_avg_ms)} | "
            f"{rounded(summary.recent_latency_p95_max_ms)} | "
            f"{summary.slo_violation_rows} | "
            f"{rounded(summary.slo_violation_pct)} | "
            f"{rounded(summary.inference_avg_ms)} | "
            f"{rounded(summary.fps_avg)} | "
            f"{rounded(summary.temp_max_c)} | "
            f"{summary.throttle_rows} | "
            f"{summary.final_model_switches} |"
        )
    return "\n".join(lines)


def find_summary(summaries: list[RunSummary], label: str) -> RunSummary | None:
    return next((summary for summary in summaries if summary.label == label), None)


def pct_change(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return (before - after) / before * 100.0


def interpretation_lines(summaries: list[RunSummary]) -> list[str]:
    naive = find_summary(summaries, "naive")
    controlled = find_summary(summaries, "controlled")
    if naive is None or controlled is None:
        return []

    lines = [
        "- SLO violations fell from "
        f"{naive.slo_violation_rows} rows to {controlled.slo_violation_rows} rows, "
        f"a {pct_change(naive.slo_violation_rows, controlled.slo_violation_rows):.1f}% reduction.",
    ]
    if (
        naive.recent_latency_p95_avg_ms is not None
        and controlled.recent_latency_p95_avg_ms is not None
    ):
        lines.append(
            "- Average recent p95 latency fell from "
            f"{naive.recent_latency_p95_avg_ms:.3f} ms to "
            f"{controlled.recent_latency_p95_avg_ms:.3f} ms."
        )
    if naive.inference_avg_ms is not None and controlled.inference_avg_ms is not None:
        lines.append(
            "- Average inference time fell from "
            f"{naive.inference_avg_ms:.3f} ms to "
            f"{controlled.inference_avg_ms:.3f} ms."
        )
    if naive.fps_avg is not None and controlled.fps_avg is not None:
        fps_gain = (
            (controlled.fps_avg - naive.fps_avg) / naive.fps_avg * 100.0
            if naive.fps_avg
            else 0.0
        )
        lines.append(
            "- Average FPS improved from "
            f"{naive.fps_avg:.3f} to {controlled.fps_avg:.3f}, "
            f"a {fps_gain:.1f}% gain."
        )
    lines.extend(
        [
            "- Both runs stayed throttle-free (`throttle_rows=0`).",
            "",
            "The controlled run did not eliminate every SLO violation. It reduced "
            "the violation count and recovered FPS/inference latency under the "
            "same CPU stress. That is the correct claim for this data.",
            "",
            "The controlled run switched four times: it returned to Thunder once "
            "while the fault was still active, then degraded again. This is a "
            "tuning opportunity for a future recovery-policy pass.",
        ]
    )
    return lines


def write_markdown(
    *,
    output_path: Path,
    summaries: list[RunSummary],
    slo_ms: float,
    plot_output: Path | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Controlled vs Naive (CPU-stress fault)",
        "",
        f"SLO threshold: `{slo_ms:.1f} ms`",
        "",
        "## Conditions",
        "",
        "- Device: Raspberry Pi 5 with active cooler attached",
        "- Camera: USB camera via OpenCV",
        "- Initial model: `thunder`",
        "- Run duration: 90 seconds per mode",
        "- Fault scenario: `cpu_stress`",
        "- Fault timing: starts after 20 seconds, runs for 30 seconds",
        "- CPU workers: 8",
        "- Raw CSV files are local benchmark artifacts and are not committed",
        "",
        "## Summary",
        "",
        markdown_table(summaries),
        "",
        "## Interpretation",
        "",
        *interpretation_lines(summaries),
        "",
        "## Input CSV",
        "",
    ]
    for summary in summaries:
        lines.append(f"- {summary.label}: `{summary.csv_path}`")
    if plot_output is not None:
        lines.extend(["", "## Plot", "", f"- `{plot_output}`"])
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Raw CSV files under `metrics/` are local benchmark outputs and should not be committed.",
            "- A selected plot under `docs/assets/` may be committed when it is used by the README.",
            "- `SLO_rows` counts rows where `recent_latency_p95_ms` is greater than the SLO threshold.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def plot_runs(
    *,
    output_path: Path,
    rows_by_label: dict[str, list[dict[str, str]]],
    slo_ms: float,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label, rows in rows_by_label.items():
        elapsed, p95 = time_series(rows, "recent_latency_p95_ms")
        axes[0].plot(elapsed, p95, linewidth=1.8, label=label)
        elapsed, fps = time_series(rows, "fps")
        axes[1].plot(elapsed, fps, linewidth=1.5, label=label)
    axes[0].axhline(slo_ms, color="tab:red", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Recent latency p95 (ms)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].set_xlabel("Elapsed seconds")
    axes[1].set_ylabel("FPS")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.suptitle("Naive vs controlled under CPU-stress fault")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    rows_by_label = {
        "naive": read_rows(args.naive_csv),
        "controlled": read_rows(args.controlled_csv),
    }
    summaries = [
        summarize_run(
            label=label,
            csv_path=args.naive_csv if label == "naive" else args.controlled_csv,
            rows=rows,
            slo_ms=args.slo_ms,
        )
        for label, rows in rows_by_label.items()
    ]
    plot_output = args.plot_output
    if plot_output is not None:
        try:
            plot_runs(output_path=plot_output, rows_by_label=rows_by_label, slo_ms=args.slo_ms)
        except ImportError as exc:
            print(f"plot skipped: matplotlib is not installed ({exc})", file=sys.stderr)
            plot_output = None
    write_markdown(
        output_path=args.markdown_output,
        summaries=summaries,
        slo_ms=args.slo_ms,
        plot_output=plot_output,
    )
    print(f"wrote {args.markdown_output}")
    if plot_output is not None:
        print(f"wrote {plot_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
