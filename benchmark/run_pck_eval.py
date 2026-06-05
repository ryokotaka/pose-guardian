"""Evaluate Lightning keypoints against Thunder pseudo ground truth.

This is not an absolute pose-accuracy benchmark. It measures how far the
Lightning model drifts from Thunder on fixed local reference clips.

Usage:
    .venv/bin/python benchmark/run_pck_eval.py \
        --stride 5 \
        --json-output metrics/pck_pseudo_gt.json \
        --markdown-output docs/pck_pseudo_gt.md

Reference clips and generated JSON files are local benchmark inputs/outputs and
should not be committed. The Markdown summary can be committed when it contains
only aggregate numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_estimator import (  # noqa: E402
    KEYPOINT_NAMES,
    PoseEstimator,
    PoseEstimatorConfig,
)


DEFAULT_CLIPS = (
    ROOT / "benchmark" / "reference_clips" / "clip_still.mp4",
    ROOT / "benchmark" / "reference_clips" / "clip_slow.mp4",
    ROOT / "benchmark" / "reference_clips" / "clip_fast.mp4",
)


@dataclass(frozen=True)
class FramePck:
    eligible_keypoints: int
    correct_keypoints: int
    distances: tuple[float, ...]
    reference_avg_confidence: float
    candidate_avg_confidence: float
    per_keypoint: tuple[tuple[str, bool, bool, float | None], ...]


@dataclass(frozen=True)
class ClipPckSummary:
    clip: str
    total_frames: int
    evaluated_frames: int
    source_fps: float
    duration_s: float
    eligible_keypoints: int
    correct_keypoints: int
    pck: float | None
    mean_distance: float | None
    median_distance: float | None
    reference_avg_confidence: float | None
    candidate_avg_confidence: float | None
    per_keypoint: dict[str, dict[str, float | int | None]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Lightning vs Thunder pseudo-ground-truth PCK on reference "
            "clips."
        )
    )
    parser.add_argument(
        "clips",
        nargs="*",
        type=Path,
        help="Clip path(s). Defaults to benchmark/reference_clips/clip_*.mp4.",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.05,
        help=(
            "PCK threshold as a fraction of the normalized image diagonal. "
            "Default 0.05 means PCK@0.05."
        ),
    )
    parser.add_argument(
        "--min-reference-confidence",
        type=float,
        default=0.3,
        help="Only Thunder keypoints at or above this confidence count as visible.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Evaluate every Nth frame. Use 1 for every frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit decoded frames per clip. 0 means no limit.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Interpreter thread count passed to PoseEstimator.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("metrics/pck_pseudo_gt.json"),
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Optional Markdown summary output path.",
    )
    return parser.parse_args()


def checked_clip_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    clips = tuple(paths) if paths else DEFAULT_CLIPS
    missing = [path for path in clips if not path.exists()]
    if missing:
        for path in missing:
            print(f"ERROR: missing clip: {path}", file=sys.stderr)
        raise SystemExit(1)
    return clips


def pck_distance_threshold(threshold_ratio: float) -> float:
    if threshold_ratio <= 0:
        raise ValueError("threshold_ratio must be > 0")
    return math.sqrt(2.0) * threshold_ratio


def normalized_distances(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if reference.shape != (17, 3) or candidate.shape != (17, 3):
        raise ValueError(
            "reference and candidate keypoints must both have shape (17, 3)"
        )
    deltas = reference[:, :2] - candidate[:, :2]
    return np.sqrt(np.sum(deltas * deltas, axis=1))


def evaluate_keypoints(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    threshold_ratio: float,
    min_reference_confidence: float,
) -> FramePck:
    threshold = pck_distance_threshold(threshold_ratio)
    distances = normalized_distances(reference, candidate)
    visible = reference[:, 2] >= min_reference_confidence
    correct = (distances <= threshold) & visible
    visible_distances = tuple(float(d) for d, keep in zip(distances, visible) if keep)
    per_keypoint = tuple(
        (
            name,
            bool(is_visible),
            bool(is_correct),
            float(distance) if is_visible else None,
        )
        for name, is_visible, is_correct, distance in zip(
            KEYPOINT_NAMES, visible, correct, distances
        )
    )
    return FramePck(
        eligible_keypoints=int(np.sum(visible)),
        correct_keypoints=int(np.sum(correct)),
        distances=visible_distances,
        reference_avg_confidence=float(np.mean(reference[:, 2])),
        candidate_avg_confidence=float(np.mean(candidate[:, 2])),
        per_keypoint=per_keypoint,
    )


def summarize_frames(
    *,
    clip: str,
    total_frames: int,
    evaluated_frames: int,
    source_fps: float,
    frames: list[FramePck],
) -> ClipPckSummary:
    eligible = sum(frame.eligible_keypoints for frame in frames)
    correct = sum(frame.correct_keypoints for frame in frames)
    distances = [distance for frame in frames for distance in frame.distances]
    ref_conf = [frame.reference_avg_confidence for frame in frames]
    cand_conf = [frame.candidate_avg_confidence for frame in frames]
    per_keypoint = summarize_per_keypoint(frames)
    return ClipPckSummary(
        clip=clip,
        total_frames=total_frames,
        evaluated_frames=evaluated_frames,
        source_fps=source_fps,
        duration_s=total_frames / source_fps if source_fps > 0 else 0.0,
        eligible_keypoints=eligible,
        correct_keypoints=correct,
        pck=(correct / eligible) if eligible else None,
        mean_distance=statistics.mean(distances) if distances else None,
        median_distance=statistics.median(distances) if distances else None,
        reference_avg_confidence=statistics.mean(ref_conf) if ref_conf else None,
        candidate_avg_confidence=statistics.mean(cand_conf) if cand_conf else None,
        per_keypoint=per_keypoint,
    )


def summarize_per_keypoint(
    frames: list[FramePck],
) -> dict[str, dict[str, float | int | None]]:
    stats: dict[str, dict[str, Any]] = {
        name: {"eligible": 0, "correct": 0, "distances": []}
        for name in KEYPOINT_NAMES
    }
    for frame in frames:
        for name, visible, correct, distance in frame.per_keypoint:
            if not visible:
                continue
            item = stats[name]
            item["eligible"] += 1
            item["correct"] += int(correct)
            item["distances"].append(float(distance))

    summary: dict[str, dict[str, float | int | None]] = {}
    for name, item in stats.items():
        eligible = int(item["eligible"])
        correct = int(item["correct"])
        distances = item["distances"]
        summary[name] = {
            "eligible": eligible,
            "correct": correct,
            "pck": (correct / eligible) if eligible else None,
            "mean_distance": statistics.mean(distances) if distances else None,
        }
    return summary


def evaluate_clip(
    path: Path,
    estimator: PoseEstimator,
    *,
    stride: int,
    max_frames: int,
    threshold_ratio: float,
    min_reference_confidence: float,
) -> ClipPckSummary:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip: {path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = 0
    evaluated_frames = 0
    frame_results: list[FramePck] = []

    try:
        while True:
            if max_frames > 0 and total_frames >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            total_frames += 1
            if (total_frames - 1) % stride != 0:
                continue

            estimator.switch_model("thunder")
            reference = estimator.estimate(frame).keypoints_array
            estimator.switch_model("lightning")
            candidate = estimator.estimate(frame).keypoints_array
            frame_results.append(
                evaluate_keypoints(
                    reference,
                    candidate,
                    threshold_ratio=threshold_ratio,
                    min_reference_confidence=min_reference_confidence,
                )
            )
            evaluated_frames += 1
    finally:
        cap.release()
        estimator.switch_model("thunder")

    if evaluated_frames == 0:
        raise RuntimeError(f"no frames evaluated: {path}")

    return summarize_frames(
        clip=path.name,
        total_frames=total_frames,
        evaluated_frames=evaluated_frames,
        source_fps=source_fps,
        frames=frame_results,
    )


def aggregate_summary(clips: list[ClipPckSummary]) -> dict[str, Any]:
    eligible = sum(clip.eligible_keypoints for clip in clips)
    correct = sum(clip.correct_keypoints for clip in clips)
    weighted_distances: list[float] = []
    ref_conf: list[float] = []
    cand_conf: list[float] = []
    for clip in clips:
        if clip.mean_distance is not None:
            weighted_distances.extend(
                [clip.mean_distance] * max(1, clip.eligible_keypoints)
            )
        if clip.reference_avg_confidence is not None:
            ref_conf.append(clip.reference_avg_confidence)
        if clip.candidate_avg_confidence is not None:
            cand_conf.append(clip.candidate_avg_confidence)
    return {
        "clips": len(clips),
        "total_frames": sum(clip.total_frames for clip in clips),
        "evaluated_frames": sum(clip.evaluated_frames for clip in clips),
        "eligible_keypoints": eligible,
        "correct_keypoints": correct,
        "pck": (correct / eligible) if eligible else None,
        "mean_distance": statistics.mean(weighted_distances)
        if weighted_distances
        else None,
        "reference_avg_confidence": statistics.mean(ref_conf) if ref_conf else None,
        "candidate_avg_confidence": statistics.mean(cand_conf) if cand_conf else None,
    }


def asdict_clip(summary: ClipPckSummary) -> dict[str, Any]:
    return {
        "clip": summary.clip,
        "total_frames": summary.total_frames,
        "evaluated_frames": summary.evaluated_frames,
        "source_fps": summary.source_fps,
        "duration_s": summary.duration_s,
        "eligible_keypoints": summary.eligible_keypoints,
        "correct_keypoints": summary.correct_keypoints,
        "pck": summary.pck,
        "mean_distance": summary.mean_distance,
        "median_distance": summary.median_distance,
        "reference_avg_confidence": summary.reference_avg_confidence,
        "candidate_avg_confidence": summary.candidate_avg_confidence,
        "per_keypoint": summary.per_keypoint,
    }


def write_json(
    path: Path,
    *,
    clips: list[ClipPckSummary],
    threshold_ratio: float,
    min_reference_confidence: float,
    runtime: str,
    num_threads: int,
    elapsed_s: float,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_model": "thunder",
        "candidate_model": "lightning",
        "metric": f"PCK@{threshold_ratio:g}",
        "threshold_ratio": threshold_ratio,
        "distance_threshold_normalized": pck_distance_threshold(threshold_ratio),
        "min_reference_confidence": min_reference_confidence,
        "runtime": runtime,
        "num_threads": num_threads,
        "elapsed_s": elapsed_s,
        "aggregate": aggregate_summary(clips),
        "clips": [asdict_clip(clip) for clip in clips],
        "notes": [
            "Thunder is used as pseudo ground truth; this is not absolute accuracy.",
            "Only Thunder keypoints above min_reference_confidence are evaluated.",
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_markdown(
    path: Path,
    *,
    clips: list[ClipPckSummary],
    threshold_ratio: float,
    min_reference_confidence: float,
    json_output: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_summary(clips)
    lines = [
        "# PCK Pseudo-Ground-Truth Evaluation",
        "",
        f"Metric: `PCK@{threshold_ratio:g}`",
        "",
        "Thunder is used as pseudo ground truth and Lightning is compared against it.",
        "This does not measure absolute human-pose accuracy. It measures the",
        "localization drift introduced when the system switches to Lightning.",
        "",
        "## Conditions",
        "",
        f"- Reference model: `thunder`",
        f"- Candidate model: `lightning`",
        f"- Threshold: `{threshold_ratio:g}` of the normalized image diagonal",
        "- Normalized distance threshold: "
        f"`{fmt(pck_distance_threshold(threshold_ratio), 4)}`",
        f"- Min reference confidence: `{min_reference_confidence:g}`",
        f"- JSON output: `{json_output}`",
        "",
        "## Summary",
        "",
        "| clip | frames | evaluated | eligible keypoints | PCK | mean distance "
        "| Thunder conf | Lightning conf |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for clip in clips:
        lines.append(
            "| "
            f"{clip.clip} | "
            f"{clip.total_frames} | "
            f"{clip.evaluated_frames} | "
            f"{clip.eligible_keypoints} | "
            f"{fmt(clip.pck)} | "
            f"{fmt(clip.mean_distance, 4)} | "
            f"{fmt(clip.reference_avg_confidence)} | "
            f"{fmt(clip.candidate_avg_confidence)} |"
        )
    lines.extend(
        [
            "| "
            f"**aggregate** | "
            f"{aggregate['total_frames']} | "
            f"{aggregate['evaluated_frames']} | "
            f"{aggregate['eligible_keypoints']} | "
            f"{fmt(aggregate['pck'])} | "
            f"{fmt(aggregate['mean_distance'], 4)} | "
            f"{fmt(aggregate['reference_avg_confidence'])} | "
            f"{fmt(aggregate['candidate_avg_confidence'])} |",
            "",
            "## Notes",
            "",
            "- `PCK` counts keypoints whose Lightning coordinate is within the threshold from Thunder.",
            "- Keypoints with Thunder confidence below the configured threshold are excluded.",
            "- The raw reference clips and JSON output are local benchmark artifacts and should not be committed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def print_result(summary: ClipPckSummary) -> None:
    print(
        "RESULT "
        f"clip={summary.clip} "
        f"frames={summary.total_frames} "
        f"evaluated={summary.evaluated_frames} "
        f"eligible_keypoints={summary.eligible_keypoints} "
        f"pck={fmt(summary.pck)} "
        f"mean_distance={fmt(summary.mean_distance, 4)} "
        f"thunder_conf={fmt(summary.reference_avg_confidence)} "
        f"lightning_conf={fmt(summary.candidate_avg_confidence)}"
    )


def main() -> int:
    args = parse_args()
    if args.stride <= 0:
        print("ERROR: --stride must be >= 1", file=sys.stderr)
        return 2
    if args.max_frames < 0:
        print("ERROR: --max-frames must be >= 0", file=sys.stderr)
        return 2
    if args.min_reference_confidence < 0 or args.min_reference_confidence > 1:
        print(
            "ERROR: --min-reference-confidence must be between 0 and 1",
            file=sys.stderr,
        )
        return 2
    try:
        pck_distance_threshold(args.threshold_ratio)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    clips = checked_clip_paths(args.clips)
    estimator = PoseEstimator(
        PoseEstimatorConfig(
            initial_model="thunder",
            num_threads=args.num_threads,
        )
    )
    info = estimator.get_model_info()
    print(
        f"runtime={info['runtime']} threads={args.num_threads} "
        f"metric=PCK@{args.threshold_ratio:g} "
        f"min_reference_confidence={args.min_reference_confidence:g}"
    )

    t0 = time.perf_counter()
    summaries: list[ClipPckSummary] = []
    for path in clips:
        summary = evaluate_clip(
            path,
            estimator,
            stride=args.stride,
            max_frames=args.max_frames,
            threshold_ratio=args.threshold_ratio,
            min_reference_confidence=args.min_reference_confidence,
        )
        summaries.append(summary)
        print_result(summary)
    elapsed_s = time.perf_counter() - t0

    json_path = write_json(
        args.json_output,
        clips=summaries,
        threshold_ratio=args.threshold_ratio,
        min_reference_confidence=args.min_reference_confidence,
        runtime=str(info["runtime"]),
        num_threads=args.num_threads,
        elapsed_s=elapsed_s,
    )
    print(f"wrote {json_path}")
    if args.markdown_output:
        markdown_path = write_markdown(
            args.markdown_output,
            clips=summaries,
            threshold_ratio=args.threshold_ratio,
            min_reference_confidence=args.min_reference_confidence,
            json_output=args.json_output,
        )
        print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
