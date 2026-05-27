"""Day 5: check recorded reference clip quality with MoveNet Thunder.

Usage:
    .venv/bin/python examples/check_clips.py
    .venv/bin/python examples/check_clips.py --stride 1
    .venv/bin/python examples/check_clips.py --threshold 0.5
    .venv/bin/python examples/check_clips.py benchmark/reference_clips/clip_still.mp4

This script reads local benchmark clips, runs PoseEstimator, and reports
confidence summary statistics. It does not write outputs by default.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

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
DEFAULT_THRESHOLDS = {
    "clip_still.mp4": 0.5,
    "clip_slow.mp4": 0.5,
    "clip_fast.mp4": 0.4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check avg MoveNet confidence for reference clips."
    )
    parser.add_argument(
        "clips",
        nargs="*",
        type=Path,
        help="Clip path(s). Defaults to benchmark/reference_clips/clip_*.mp4.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Analyze every Nth frame. Use 1 for every frame.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Warn when avg_confidence is below this value. If omitted, "
            "uses per-clip defaults: still/slow=0.5, fast=0.4."
        ),
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=850,
        help="Warn when a clip has fewer decoded frames. Use 0 to disable.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show frames while checking. Off by default.",
    )
    return parser.parse_args()


def analyze_clip(
    path: Path,
    estimator: PoseEstimator,
    *,
    stride: int,
    display: bool,
) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "error": "cannot open"}

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = 0
    analyzed_frames = 0
    confs: list[float] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            total_frames += 1
            if (total_frames - 1) % stride != 0:
                continue

            pose = estimator.estimate(frame)
            confs.append(pose.avg_confidence)
            analyzed_frames += 1

            if display:
                preview = frame.copy()
                for kp in pose.keypoints:
                    if kp.confidence < 0.3:
                        continue
                    h, w = preview.shape[:2]
                    cv2.circle(
                        preview,
                        (int(kp.x * w), int(kp.y * h)),
                        4,
                        (0, 255, 0),
                        -1,
                    )
                cv2.imshow(str(path), preview)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

    if not confs:
        return {
            "path": str(path),
            "total_frames": total_frames,
            "analyzed_frames": analyzed_frames,
            "error": "no frames analyzed",
        }

    return {
        "path": str(path),
        "total_frames": total_frames,
        "analyzed_frames": analyzed_frames,
        "source_fps": source_fps,
        "duration_s": total_frames / source_fps if source_fps > 0 else 0.0,
        "avg_confidence": statistics.mean(confs),
        "median_confidence": statistics.median(confs),
        "min_confidence": min(confs),
        "max_confidence": max(confs),
    }


def threshold_for_path(path: Path, override: float | None) -> float:
    if override is not None:
        return override
    return DEFAULT_THRESHOLDS.get(path.name, 0.5)


def print_result(
    result: dict[str, object],
    *,
    threshold: float,
    min_frames: int,
) -> bool:
    path = Path(str(result["path"]))
    if "error" in result:
        print(f"{path.name}: ERROR {result['error']}")
        return False

    avg = float(result["avg_confidence"])
    total_frames = int(result["total_frames"])
    frame_count_ok = min_frames <= 0 or total_frames >= min_frames
    confidence_ok = avg >= threshold
    ok = confidence_ok and frame_count_ok
    status = "OK" if ok else "WARN"
    reasons = []
    if not confidence_ok:
        reasons.append(f"avg<{threshold:.2f}")
    if not frame_count_ok:
        reasons.append(f"frames<{min_frames}")
    reason_text = f" reason={','.join(reasons)}" if reasons else ""
    print(
        f"{path.name}: {status} "
        f"frames={total_frames} "
        f"duration={float(result['duration_s']):.1f}s "
        f"fps={float(result['source_fps']):.2f} "
        f"analyzed={result['analyzed_frames']} "
        f"avg={avg:.3f} "
        f"median={float(result['median_confidence']):.3f} "
        f"min={float(result['min_confidence']):.3f} "
        f"max={float(result['max_confidence']):.3f} "
        f"threshold={threshold:.2f}"
        f"{reason_text}"
    )
    return ok


def main() -> int:
    args = parse_args()
    if args.stride <= 0:
        print("ERROR: --stride must be >= 1", file=sys.stderr)
        return 2

    clip_paths = tuple(args.clips) if args.clips else DEFAULT_CLIPS
    missing = [path for path in clip_paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"ERROR: missing clip: {path}", file=sys.stderr)
        return 1

    estimator = PoseEstimator(
        PoseEstimatorConfig(initial_model="thunder")
    )

    all_ok = True
    for path in clip_paths:
        result = analyze_clip(
            path,
            estimator,
            stride=args.stride,
            display=args.display,
        )
        all_ok = print_result(
            result,
            threshold=threshold_for_path(path, args.threshold),
            min_frames=args.min_frames,
        ) and all_ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
