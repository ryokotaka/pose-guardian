"""Run live pose inference with ResourceController actions applied.

Usage:
    .venv/bin/python examples/run_controlled.py --no-display --duration 70
    .venv/bin/python examples/run_controlled.py --device 0 --duration 70

Generated CSV files and plots are local benchmark outputs under ``metrics/``;
do not commit them.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Sequence
import csv
from dataclasses import dataclass
import gc
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

from examples.run_monitored import (  # noqa: E402
    CSV_FIELDS as MONITORED_CSV_FIELDS,
    csv_row as monitored_csv_row,
    default_plot_path,
    maybe_plot,
    plot_metrics,
    rounded,
    validate_args as validate_monitored_args,
)
from examples.sanity_check import draw_pose  # noqa: E402
from src.camera import Camera, CameraConfig  # noqa: E402
from src.fault_injector import FaultInjector, FaultScenario  # noqa: E402
from src.pose_estimator import (  # noqa: E402
    PoseEstimator,
    PoseEstimatorConfig,
    PoseResult,
)
from src.resource_controller import (  # noqa: E402
    ActionType,
    ControlAction,
    ControllerState,
    ResourceController,
)
from src.resource_monitor import (  # noqa: E402
    MonitorConfig,
    ResourceMonitor,
    ResourceSnapshot,
)


CONTROL_CSV_FIELDS = [
    "state",
    "previous_state",
    "action",
    "action_reason",
    "skipped_frames",
    "model_switches",
    "force_gc_count",
    "switch_ms",
    "recent_latency_p95_ms",
]
FAULT_CSV_FIELDS = [
    "fault_scenario",
    "fault_active",
]
CSV_FIELDS = [*MONITORED_CSV_FIELDS, *CONTROL_CSV_FIELDS, *FAULT_CSV_FIELDS]


@dataclass
class ControlRuntimeStats:
    skipped_frames: int = 0
    model_switches: int = 0
    force_gc_count: int = 0
    last_switch_ms: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live camera inference with ResourceController actions."
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
        help="Initial model.",
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
        default=Path("metrics/week3_day3_run_controlled.csv"),
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
        "--latency-window-size",
        type=int,
        default=60,
        help="Recent inference latency count passed to ResourceController.",
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
    parser.add_argument(
        "--fault-scenario",
        choices=[scenario.value for scenario in FaultScenario],
        default=FaultScenario.NONE.value,
        help="Optional pressure scenario to start during the run.",
    )
    parser.add_argument(
        "--fault-start-after",
        type=float,
        default=10.0,
        help="Seconds after start before fault injection begins.",
    )
    parser.add_argument(
        "--fault-duration",
        type=float,
        default=20.0,
        help="Fault injection duration in seconds.",
    )
    parser.add_argument(
        "--fault-memory-target-percent",
        type=float,
        default=85.0,
        help="Memory pressure target percent. Hard-capped by FaultInjector.",
    )
    parser.add_argument(
        "--fault-cpu-workers",
        type=int,
        default=2,
        help="CPU workers for cpu_stress scenario.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> int:
    validation = validate_monitored_args(args)
    if validation:
        return validation
    if args.latency_window_size <= 0:
        print("ERROR: --latency-window-size must be > 0", file=sys.stderr)
        return 2
    if args.fault_start_after < 0:
        print("ERROR: --fault-start-after must be >= 0", file=sys.stderr)
        return 2
    if args.fault_duration <= 0:
        print("ERROR: --fault-duration must be > 0", file=sys.stderr)
        return 2
    if args.fault_cpu_workers <= 0:
        print("ERROR: --fault-cpu-workers must be > 0", file=sys.stderr)
        return 2
    return 0


def latency_p95(values: Sequence[float]) -> float | None:
    clean = sorted(float(value) for value in values if value >= 0)
    if not clean:
        return None
    index = int(np.ceil(0.95 * len(clean))) - 1
    return clean[max(0, min(index, len(clean) - 1))]


def apply_control_action(
    action: ControlAction,
    estimator: Any,
    stats: ControlRuntimeStats,
) -> None:
    stats.last_switch_ms = 0.0
    if action.action is ActionType.SWITCH_TO_LIGHT:
        if estimator.current_model() != "lightning":
            stats.last_switch_ms = estimator.switch_model("lightning")
            stats.model_switches += 1
    elif action.action is ActionType.SWITCH_TO_HEAVY:
        if estimator.current_model() != "thunder":
            stats.last_switch_ms = estimator.switch_model("thunder")
            stats.model_switches += 1
    elif action.action is ActionType.FORCE_GC:
        gc.collect()
        stats.force_gc_count += 1


def should_skip_frame(
    *,
    state: ControllerState,
    frame_sequence: int,
    frame_skip_ratio: float,
) -> bool:
    if state is not ControllerState.CRITICAL:
        return False
    if frame_skip_ratio <= 0:
        return False
    if frame_skip_ratio >= 1:
        return True
    period = max(2, round(1.0 / frame_skip_ratio))
    return frame_sequence % period == 0


def controlled_csv_row(
    *,
    snapshot: ResourceSnapshot,
    elapsed_s: float,
    model_name: str,
    frames_processed: int,
    last_pose: PoseResult | None,
    control_action: ControlAction | None,
    stats: ControlRuntimeStats,
    recent_latencies_ms: Sequence[float],
    fault_scenario: FaultScenario = FaultScenario.NONE,
    fault_active: bool = False,
) -> dict[str, Any]:
    row = monitored_csv_row(
        snapshot=snapshot,
        elapsed_s=elapsed_s,
        model_name=model_name,
        frames_processed=frames_processed,
        last_pose=last_pose,
    )
    row.update(
        {
            "state": (
                control_action.state.value
                if control_action is not None
                else ControllerState.NORMAL.value
            ),
            "previous_state": (
                control_action.previous_state.value if control_action is not None else ""
            ),
            "action": (
                control_action.action.value
                if control_action is not None
                else ActionType.NONE.value
            ),
            "action_reason": control_action.reason if control_action is not None else "",
            "skipped_frames": stats.skipped_frames,
            "model_switches": stats.model_switches,
            "force_gc_count": stats.force_gc_count,
            "switch_ms": rounded(stats.last_switch_ms),
            "recent_latency_p95_ms": rounded(latency_p95(recent_latencies_ms)),
            "fault_scenario": fault_scenario.value,
            "fault_active": fault_active,
        }
    )
    return row


def overlay_control_stats(
    frame: np.ndarray,
    *,
    action: ControlAction,
    stats: ControlRuntimeStats,
) -> None:
    height, width = frame.shape[:2]
    y = max(64, height - 42)
    cv2.rectangle(frame, (0, y - 20), (width, y + 18), (0, 0, 0), -1)
    line = (
        f"state {action.state.value} | action {action.action.value} | "
        f"skipped {stats.skipped_frames} | switches {stats.model_switches}"
    )
    cv2.putText(
        frame,
        line,
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def write_controlled_row(
    *,
    writer: csv.DictWriter,
    snapshot: ResourceSnapshot,
    elapsed_s: float,
    model_name: str,
    frames_done: int,
    last_pose: PoseResult | None,
    control_action: ControlAction | None,
    stats: ControlRuntimeStats,
    recent_latencies_ms: Sequence[float],
    fault_scenario: FaultScenario = FaultScenario.NONE,
    fault_active: bool = False,
) -> None:
    writer.writerow(
        controlled_csv_row(
            snapshot=snapshot,
            elapsed_s=elapsed_s,
            model_name=model_name,
            frames_processed=frames_done,
            last_pose=last_pose,
            control_action=control_action,
            stats=stats,
            recent_latencies_ms=recent_latencies_ms,
            fault_scenario=fault_scenario,
            fault_active=fault_active,
        )
    )


def start_configured_fault(args: argparse.Namespace, injector: FaultInjector) -> None:
    scenario = FaultScenario(args.fault_scenario)
    if scenario is FaultScenario.NONE:
        return
    if scenario is FaultScenario.MEMORY_PRESSURE:
        injector.inject_memory_pressure(
            target_percent=args.fault_memory_target_percent,
            duration_sec=args.fault_duration,
        )
    elif scenario is FaultScenario.CPU_STRESS:
        injector.inject_cpu_stress(
            duration_sec=args.fault_duration,
            num_workers=args.fault_cpu_workers,
        )
    elif scenario is FaultScenario.CAMERA_DISCONNECT:
        raise ValueError("camera_disconnect is not wired into Camera in Week3 Day4")
    else:
        raise ValueError(f"unsupported fault scenario: {scenario.value}")


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
    controller = ResourceController()
    fault_scenario = FaultScenario(args.fault_scenario)
    fault_injector = FaultInjector()
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
    print(f"controller state: {controller.state.value}")
    print(f"logging CSV: {args.csv_output}")
    if fault_scenario is not FaultScenario.NONE:
        print(
            f"fault: {fault_scenario.value} starts after "
            f"{args.fault_start_after:.1f}s for {args.fault_duration:.1f}s"
        )
    if not args.no_display:
        print("keys: 'q' quit")

    window_name = "Edge Inference Guardian controlled"
    if not args.no_display:
        dummy = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        cv2.imshow(window_name, dummy)
        cv2.waitKey(1)

    frames_seen = 0
    frames_done = 0
    last_frame_id = 0
    last_pose: PoseResult | None = None
    last_action: ControlAction | None = None
    recent_latencies: deque[float] = deque(maxlen=args.latency_window_size)
    stats = ControlRuntimeStats()
    start_t = time.perf_counter()
    next_log_t = start_t
    rows_written = 0
    fault_started = False
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
                if (
                    fault_scenario is not FaultScenario.NONE
                    and not fault_started
                    and elapsed >= args.fault_start_after
                ):
                    try:
                        start_configured_fault(args, fault_injector)
                    except ValueError as exc:
                        print(f"ERROR: {exc}", file=sys.stderr)
                        return 2
                    fault_started = True
                    print(
                        f"fault started: {fault_scenario.value} "
                        f"duration={args.fault_duration:.1f}s"
                    )
                fault_active = fault_injector.is_active(fault_scenario)

                frame, current_id = cam.read_new_frame(last_frame_id)
                if frame is None:
                    if now >= next_log_t:
                        write_controlled_row(
                            writer=writer,
                            snapshot=monitor.snapshot(),
                            elapsed_s=elapsed,
                            model_name=estimator.current_model(),
                            frames_done=frames_done,
                            last_pose=last_pose,
                            control_action=last_action,
                            stats=stats,
                            recent_latencies_ms=tuple(recent_latencies),
                            fault_scenario=fault_scenario,
                            fault_active=fault_active,
                        )
                        f.flush()
                        rows_written += 1
                        next_log_t += args.log_interval
                    if not args.no_display and (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                    time.sleep(0.005)
                    continue

                last_frame_id = current_id
                frames_seen += 1
                snapshot = monitor.snapshot()
                action = controller.evaluate(snapshot, tuple(recent_latencies))
                last_action = action
                apply_control_action(action, estimator, stats)

                skip_frame = should_skip_frame(
                    state=action.state,
                    frame_sequence=frames_seen,
                    frame_skip_ratio=controller.config.frame_skip_ratio,
                )
                if skip_frame:
                    stats.skipped_frames += 1
                else:
                    pose = estimator.estimate(frame)
                    last_pose = pose
                    frames_done += 1
                    recent_latencies.append(pose.inference_time_ms)
                    monitor.record_inference(pose.inference_time_ms)
                    snapshot = monitor.snapshot()

                now = time.perf_counter()
                elapsed = now - start_t
                if now >= next_log_t:
                    write_controlled_row(
                        writer=writer,
                        snapshot=snapshot,
                        elapsed_s=elapsed,
                        model_name=estimator.current_model(),
                        frames_done=frames_done,
                        last_pose=last_pose,
                        control_action=action,
                        stats=stats,
                        recent_latencies_ms=tuple(recent_latencies),
                        fault_scenario=fault_scenario,
                        fault_active=fault_active,
                    )
                    f.flush()
                    rows_written += 1
                    next_log_t += args.log_interval
                    while next_log_t <= now:
                        next_log_t += args.log_interval

                if not args.no_display:
                    if last_pose is not None:
                        overlay = draw_pose(frame, last_pose.keypoints_array, args.threshold)
                    else:
                        overlay = frame.copy()
                    overlay_control_stats(overlay, action=action, stats=stats)
                    cv2.imshow(window_name, overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        fault_injector.clear_all()
        monitor.stop()
        cam.stop()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_t
    print(
        f"finished: elapsed={elapsed:.1f}s  frames={frames_done}  "
        f"skipped={stats.skipped_frames}  switches={stats.model_switches}  "
        f"force_gc={stats.force_gc_count}  rows={rows_written}  csv={args.csv_output}"
    )
    print(f"camera stats: {cam.stats()}")
    if not args.no_plot:
        maybe_plot(args.csv_output, args.plot_output or default_plot_path(args.csv_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
