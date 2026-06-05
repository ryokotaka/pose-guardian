"""Record local reference clips with a USB webcam.

Usage:
    .venv/bin/python examples/record_clips.py
    .venv/bin/python examples/record_clips.py --device 1
    .venv/bin/python examples/record_clips.py --duration 5 --clip still

The generated mp4 files are written under benchmark/reference_clips/ and are
ignored by Git. Keep these clips local; they are benchmark inputs, not public
demo assets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


CLIPS = ("still", "slow", "fast")
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmark" / "reference_clips"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record 30s reference clips for benchmark input."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="OpenCV camera device index. Try 1 if 0 opens the wrong camera.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Seconds per clip.",
    )
    parser.add_argument(
        "--clip",
        choices=CLIPS,
        help="Record only one clip instead of all three.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for clip_*.mp4 outputs.",
    )
    return parser.parse_args()


def open_capture(device: int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"cannot open camera device {device}. On macOS, grant camera "
            "permission to the terminal/editor running Python, then restart it."
        )
    return cap


def countdown() -> None:
    for value in (3, 2, 1):
        print(value, flush=True)
        time.sleep(1.0)


def record_clip(
    name: str,
    *,
    device: int,
    width: int,
    height: int,
    fps: float,
    duration: float,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"clip_{name}.mp4"

    cap = open_capture(device, width, height, fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video writer: {output_path}")

    print(f"\n=== {name}: {duration:.0f}s ===")
    print("Preview window opens. Press 'q' to abort this clip.")
    countdown()
    print("recording!", flush=True)

    start = time.perf_counter()
    frames = 0
    try:
        while (time.perf_counter() - start) < duration:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            writer.write(frame)
            frames += 1

            preview = frame.copy()
            cv2.putText(
                preview,
                f"REC {name}  {frames} frames",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(f"recording {name}", preview)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    print(
        f"wrote {output_path}  frames={frames}  "
        f"elapsed={elapsed:.1f}s  effective_fps={frames / max(elapsed, 1e-6):.2f}"
    )
    return output_path


def main() -> int:
    args = parse_args()
    clip_names = (args.clip,) if args.clip else CLIPS

    for index, name in enumerate(clip_names, start=1):
        if name == "still":
            print("\nstill: stay mostly still; small breathing or posture shifts are OK.")
        elif name == "slow":
            print("\nslow: move arms and torso slowly, keep full body in frame.")
        elif name == "fast":
            print("\nfast: faster arm/body movement, but keep the camera stable.")

        try:
            record_clip(
                name,
                device=args.device,
                width=args.width,
                height=args.height,
                fps=args.fps,
                duration=args.duration,
                output_dir=args.output_dir,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        if index < len(clip_names):
            input("Done. Press Enter when ready for the next clip...")

    print("\nDone. Reference clips are local benchmark inputs; do not commit mp4 files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
