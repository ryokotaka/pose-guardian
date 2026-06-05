"""Benchmark MoveNet inference latency on fixed reference clips.

Usage:
    .venv/bin/python examples/benchmark_clips.py
    .venv/bin/python examples/benchmark_clips.py --models thunder
    .venv/bin/python examples/benchmark_clips.py --max-frames 60
    .venv/bin/python examples/benchmark_clips.py --csv-output metrics/fixed_clip_mac.csv

The default run processes every frame of the three local reference clips with
both Thunder and Lightning. Clip files and generated CSVs are local benchmark
inputs/outputs and should not be committed.
"""

from __future__ import annotations

import argparse
import csv
import platform
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_estimator import PoseEstimator, PoseEstimatorConfig  # noqa: E402


DEFAULT_CLIPS = (
    ROOT / "benchmark" / "reference_clips" / "clip_still.mp4",
    ROOT / "benchmark" / "reference_clips" / "clip_slow.mp4",
    ROOT / "benchmark" / "reference_clips" / "clip_fast.mp4",
)
DEFAULT_MODELS = ("thunder", "lightning")


@dataclass(frozen=True)
class BenchmarkResult:
    host: str
    platform: str
    runtime: str
    clip: str
    model: str
    frames: int
    source_fps: float
    wall_time_s: float
    effective_fps: float
    avg_confidence: float
    preprocess_avg_ms: float
    preprocess_median_ms: float
    preprocess_p95_ms: float
    inference_avg_ms: float
    inference_median_ms: float
    inference_p95_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PoseEstimator on fixed mp4 reference clips."
    )
    parser.add_argument(
        "clips",
        nargs="*",
        type=Path,
        help="Clip path(s). Defaults to benchmark/reference_clips/clip_*.mp4.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
        help="Model(s) to run. Defaults to both Thunder and Lightning.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit decoded frames per clip/model. 0 means all frames.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Run this many untimed estimates on the first frame before timing.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Interpreter thread count passed to PoseEstimatorConfig.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="Optional CSV output path. Parent directories are created.",
    )
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def checked_clip_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    clips = tuple(paths) if paths else DEFAULT_CLIPS
    missing = [path for path in clips if not path.exists()]
    if missing:
        for path in missing:
            print(f"ERROR: missing clip: {path}", file=sys.stderr)
        raise SystemExit(1)
    return clips


def read_first_frame(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open clip: {path}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"cannot read first frame: {path}")
        return frame
    finally:
        cap.release()


def benchmark_clip(
    path: Path,
    estimator: PoseEstimator,
    *,
    model: str,
    max_frames: int,
    warmup_frames: int,
) -> BenchmarkResult:
    estimator.switch_model(model)
    first_frame = read_first_frame(path)
    for _ in range(max(0, warmup_frames)):
        estimator.estimate(first_frame)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip: {path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    confidences: list[float] = []
    frames = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames += 1
            pose = estimator.estimate(frame)
            preprocess_ms.append(pose.preprocess_time_ms)
            inference_ms.append(pose.inference_time_ms)
            confidences.append(pose.avg_confidence)
            if max_frames > 0 and frames >= max_frames:
                break
    finally:
        cap.release()
    wall_time_s = time.perf_counter() - t0

    if frames == 0:
        raise RuntimeError(f"no frames decoded: {path}")

    info = estimator.get_model_info()
    host = socket.gethostname()
    return BenchmarkResult(
        host=host,
        platform=platform.platform(),
        runtime=str(info["runtime"]),
        clip=path.name,
        model=model,
        frames=frames,
        source_fps=source_fps,
        wall_time_s=wall_time_s,
        effective_fps=frames / wall_time_s if wall_time_s > 0 else 0.0,
        avg_confidence=statistics.mean(confidences),
        preprocess_avg_ms=statistics.mean(preprocess_ms),
        preprocess_median_ms=statistics.median(preprocess_ms),
        preprocess_p95_ms=percentile(preprocess_ms, 95.0),
        inference_avg_ms=statistics.mean(inference_ms),
        inference_median_ms=statistics.median(inference_ms),
        inference_p95_ms=percentile(inference_ms, 95.0),
    )


def print_result(result: BenchmarkResult) -> None:
    print(
        "RESULT "
        f"host={result.host} "
        f"clip={result.clip} "
        f"model={result.model} "
        f"frames={result.frames} "
        f"source_fps={result.source_fps:.2f} "
        f"effective_fps={result.effective_fps:.2f} "
        f"inference_avg_ms={result.inference_avg_ms:.2f} "
        f"inference_median_ms={result.inference_median_ms:.2f} "
        f"inference_p95_ms={result.inference_p95_ms:.2f} "
        f"preprocess_avg_ms={result.preprocess_avg_ms:.2f} "
        f"preprocess_median_ms={result.preprocess_median_ms:.2f} "
        f"preprocess_p95_ms={result.preprocess_p95_ms:.2f} "
        f"avg_confidence={result.avg_confidence:.3f}"
    )


def write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(BenchmarkResult.__dataclass_fields__.keys()),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)
    print(f"wrote {path}")


def main() -> int:
    args = parse_args()
    if args.max_frames < 0:
        print("ERROR: --max-frames must be >= 0", file=sys.stderr)
        return 2
    if args.warmup_frames < 0:
        print("ERROR: --warmup-frames must be >= 0", file=sys.stderr)
        return 2

    clips = checked_clip_paths(args.clips)
    estimator = PoseEstimator(
        PoseEstimatorConfig(
            initial_model=args.models[0],
            num_threads=args.num_threads,
        )
    )
    info = estimator.get_model_info()
    print(
        f"runtime={info['runtime']} "
        f"threads={args.num_threads} "
        f"host={socket.gethostname()} "
        f"platform={platform.platform()}"
    )

    results: list[BenchmarkResult] = []
    for clip in clips:
        for model in args.models:
            result = benchmark_clip(
                clip,
                estimator,
                model=model,
                max_frames=args.max_frames,
                warmup_frames=args.warmup_frames,
            )
            results.append(result)
            print_result(result)

    if args.csv_output:
        write_csv(args.csv_output, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
