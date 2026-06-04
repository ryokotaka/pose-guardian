"""Run live pose inference while logging resource snapshots to CSV.

Usage:
    .venv/bin/python examples/run_monitored.py --no-display --duration 70
    .venv/bin/python examples/run_monitored.py --device 0 --model thunder --duration 70
    .venv/bin/python examples/run_monitored.py --plot-only metrics/day4_pi.csv

Generated CSV files and plots are local benchmark outputs under ``metrics/``;
do not commit them.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow running this file directly (without ``pip install -e .``).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.sanity_check import draw_pose  # noqa: E402
from src.camera import Camera, CameraConfig  # noqa: E402
from src.pose_estimator import (  # noqa: E402
    PoseEstimator,
    PoseEstimatorConfig,
    PoseResult,
)
from src.resource_monitor import (  # noqa: E402
    MonitorConfig,
    ResourceMonitor,
    ResourceSnapshot,
)


CSV_FIELDS = [
    "timestamp",
    "elapsed_s",
    "model",
    "frames_processed",
    "inference_ms",
    "preprocess_ms",
    "avg_confidence",
    "cpu_temp",
    "cpu_usage",
    "cpu_freq",
    "mem_percent",
    "fps",
    "is_throttled",
    "throttle_flags",
    "pmic_rail_estimate_watts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live camera inference with ResourceMonitor CSV logging."
    )
    parser.add_argument(
        "--thunder",
        type=Path,
        default=Path("models/movenet_thunder.tflite"),
    )
    parser.add_argument(
        "--lightning",
        type=Path,
        default=Path("models/movenet_lightning.tflite"),
    )
    parser.add_argument(
        "--model",
        choices=["thunder", "lightning"],
        default="thunder",
        help="Initial model. In display mode, press t/l to switch.",
    )
    parser.add_argument("--device", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--duration",
        type=float,
        default=70.0,
        help="Auto-quit after N seconds. Use 0 to run until q.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening a GUI window.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("metrics/day4_run_monitored.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Optional PNG output path. Defaults to metrics/plots/<csv-stem>.png.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation after the run.",
    )
    parser.add_argument(
        "--plot-only",
        type=Path,
        help="Create a plot from an existing CSV and exit.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="ResourceMonitor sampling interval in seconds.",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=1.0,
        help="CSV row interval in seconds.",
    )
    parser.add_argument(
        "--fps-window-size",
        type=int,
        default=30,
        help="Frame count used for rolling FPS.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Interpreter thread count passed to PoseEstimator.",
    )
    parser.add_argument(
        "--disable-power",
        action="store_true",
        help="Do not run vcgencmd pmic_read_adc for PMIC rail estimates.",
    )
    return parser.parse_args()


def default_plot_path(csv_path: Path) -> Path:
    return csv_path.parent / "plots" / f"{csv_path.stem}.png"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def rounded(value: float | None, digits: int = 3) -> float | str:
    if value is None:
        return ""
    return round(float(value), digits)


def csv_row(
    *,
    snapshot: ResourceSnapshot,
    elapsed_s: float,
    model_name: str,
    frames_processed: int,
    last_pose: PoseResult | None,
) -> dict[str, Any]:
    return {
        "timestamp": utc_timestamp(),
        "elapsed_s": rounded(elapsed_s),
        "model": model_name,
        "frames_processed": frames_processed,
        "inference_ms": (
            rounded(last_pose.inference_time_ms) if last_pose is not None else ""
        ),
        "preprocess_ms": (
            rounded(last_pose.preprocess_time_ms) if last_pose is not None else ""
        ),
        "avg_confidence": (
            rounded(last_pose.avg_confidence) if last_pose is not None else ""
        ),
        "cpu_temp": rounded(snapshot.cpu_temp_celsius),
        "cpu_usage": rounded(snapshot.cpu_usage_percent),
        "cpu_freq": rounded(snapshot.cpu_freq_mhz),
        "mem_percent": rounded(snapshot.memory_used_percent),
        "fps": rounded(snapshot.fps),
        "is_throttled": snapshot.is_throttled,
        "throttle_flags": hex(snapshot.throttle_flags),
        "pmic_rail_estimate_watts": rounded(snapshot.pmic_rail_estimate_watts),
    }


def overlay_monitor_stats(
    frame: np.ndarray,
    *,
    pose: PoseResult,
    snapshot: ResourceSnapshot,
) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 46), (0, 0, 0), -1)
    line1 = (
        f"{pose.model_name:9s} | "
        f"infer {pose.inference_time_ms:5.1f}ms | "
        f"pre {pose.preprocess_time_ms:4.1f}ms | "
        f"conf {pose.avg_confidence:.2f}"
    )
    line2 = (
        f"temp {snapshot.cpu_temp_celsius:4.1f}C | "
        f"fps {snapshot.fps:4.1f} | "
        f"mem {snapshot.memory_used_percent:4.1f}% | "
        f"throttle {hex(snapshot.throttle_flags)}"
    )
    cv2.putText(
        frame,
        line1,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        line2,
        (8, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def plot_metrics(csv_path: Path, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    elapsed: list[float] = []
    temps: list[float] = []
    fps_values: list[float] = []
    mem_values: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = parse_float(row.get("elapsed_s", ""))
            temp = parse_float(row.get("cpu_temp", ""))
            fps = parse_float(row.get("fps", ""))
            mem = parse_float(row.get("mem_percent", ""))
            if t is None or temp is None:
                continue
            elapsed.append(t)
            temps.append(temp)
            fps_values.append(fps if fps is not None else 0.0)
            mem_values.append(mem if mem is not None else 0.0)

    if not elapsed:
        raise ValueError(f"no plottable rows in {csv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(elapsed, temps, color="tab:red", linewidth=1.8)
    axes[0].set_ylabel("CPU temp (C)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(elapsed, fps_values, color="tab:blue", linewidth=1.8, label="FPS")
    axes[1].plot(
        elapsed,
        mem_values,
        color="tab:green",
        linewidth=1.5,
        label="Memory %",
    )
    axes[1].set_xlabel("Elapsed seconds")
    axes[1].set_ylabel("FPS / Memory %")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.suptitle(csv_path.name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def maybe_plot(csv_path: Path, output_path: Path) -> None:
    try:
        path = plot_metrics(csv_path, output_path)
    except ImportError as exc:
        print(f"plot skipped: matplotlib is not installed ({exc})")
    except ValueError as exc:
        print(f"plot skipped: {exc}")
    else:
        print(f"wrote plot {path}")


def validate_args(args: argparse.Namespace) -> int:
    if args.duration < 0:
        print("ERROR: --duration must be >= 0", file=sys.stderr)
        return 2
    if args.sample_interval <= 0:
        print("ERROR: --sample-interval must be > 0", file=sys.stderr)
        return 2
    if args.log_interval <= 0:
        print("ERROR: --log-interval must be > 0", file=sys.stderr)
        return 2
    if args.fps_window_size <= 0:
        print("ERROR: --fps-window-size must be > 0", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    args = parse_args()
    validation = validate_args(args)
    if validation:
        return validation

    plot_output = args.plot_output or default_plot_path(args.plot_only or args.csv_output)
    if args.plot_only:
        try:
            path = plot_metrics(args.plot_only, plot_output)
        except (ImportError, ValueError) as exc:
            print(f"ERROR: unable to plot {args.plot_only}: {exc}", file=sys.stderr)
            return 1
        print(f"wrote plot {path}")
        return 0

    estimator = PoseEstimator(
        PoseEstimatorConfig(
            heavy_model_path=args.thunder,
            light_model_path=args.lightning,
            initial_model=args.model,
            num_threads=args.num_threads,
        )
    )
    info = estimator.get_model_info()
    monitor = ResourceMonitor(
        MonitorConfig(
            sample_interval_sec=args.sample_interval,
            fps_window_size=args.fps_window_size,
            enable_power=not args.disable_power,
        )
    )
    cam = Camera(
        CameraConfig(
            device_index=args.device,
            width=args.width,
            height=args.height,
            fps_cap=30,
        )
    )

    try:
        cam.start()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"runtime: {info['runtime']}  model: {estimator.current_model()}  "
        f"threads: {args.num_threads}"
    )
    print(f"logging CSV: {args.csv_output}")
    if not args.no_display:
        print("keys: 't' Thunder | 'l' Lightning | 'q' quit")

    window_name = "Edge Inference Guardian monitored"
    if not args.no_display:
        dummy = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        cv2.imshow(window_name, dummy)
        cv2.waitKey(1)

    frames_done = 0
    last_frame_id = 0
    last_pose: PoseResult | None = None
    start_t = time.perf_counter()
    next_log_t = start_t
    rows_written = 0
    monitor.start()

    try:
        with args.csv_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

            while True:
                now = time.perf_counter()
                elapsed = now - start_t
                if args.duration > 0 and elapsed >= args.duration:
                    break

                frame, current_id = cam.read_new_frame(last_frame_id)
                if frame is None:
                    if now >= next_log_t:
                        writer.writerow(
                            csv_row(
                                snapshot=monitor.snapshot(),
                                elapsed_s=elapsed,
                                model_name=estimator.current_model(),
                                frames_processed=frames_done,
                                last_pose=last_pose,
                            )
                        )
                        f.flush()
                        rows_written += 1
                        next_log_t += args.log_interval
                    if not args.no_display and (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                    time.sleep(0.005)
                    continue

                last_frame_id = current_id
                pose = estimator.estimate(frame)
                last_pose = pose
                frames_done += 1
                monitor.record_inference(pose.inference_time_ms)

                snapshot = monitor.snapshot()
                now = time.perf_counter()
                elapsed = now - start_t
                if now >= next_log_t:
                    writer.writerow(
                        csv_row(
                            snapshot=snapshot,
                            elapsed_s=elapsed,
                            model_name=pose.model_name,
                            frames_processed=frames_done,
                            last_pose=pose,
                        )
                    )
                    f.flush()
                    rows_written += 1
                    next_log_t += args.log_interval
                    while next_log_t <= now:
                        next_log_t += args.log_interval

                if not args.no_display:
                    overlay = draw_pose(frame, pose.keypoints_array, args.threshold)
                    overlay_monitor_stats(overlay, pose=pose, snapshot=snapshot)
                    cv2.imshow(window_name, overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("t"):
                        estimator.switch_model("thunder")
                    if key == ord("l"):
                        estimator.switch_model("lightning")
    finally:
        monitor.stop()
        cam.stop()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_t
    print(
        f"finished: elapsed={elapsed:.1f}s  frames={frames_done}  "
        f"rows={rows_written}  csv={args.csv_output}"
    )
    print(f"camera stats: {cam.stats()}")
    if not args.no_plot:
        maybe_plot(args.csv_output, args.plot_output or default_plot_path(args.csv_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
