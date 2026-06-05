"""Real-time pose estimation demo via PoseEstimator.

Usage:
    .venv/bin/python examples/run_demo.py
    .venv/bin/python examples/run_demo.py --duration 30
    .venv/bin/python examples/run_demo.py --no-display --duration 5
    .venv/bin/python examples/run_demo.py --initial lightning --duration 5

Keys (when --no-display is NOT set):
    t: switch to Thunder (heavy)
    l: switch to Lightning (light)
    q: quit

Pose estimation is encapsulated in ``src.pose_estimator.PoseEstimator``.
Drawing reuses ``examples.sanity_check.draw_pose``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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


def overlay_stats(
    frame: np.ndarray,
    pose: PoseResult,
    fps: float,
) -> None:
    """Draw a black status bar with model / latency / FPS / confidence."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    text = (
        f"{pose.model_name:9s} | "
        f"infer {pose.inference_time_ms:5.1f}ms | "
        f"pre {pose.preprocess_time_ms:4.1f}ms | "
        f"{fps:4.1f}FPS | conf {pose.avg_confidence:.2f}"
    )
    cv2.putText(
        frame,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time MoveNet demo on local webcam.")
    p.add_argument(
        "--thunder",
        type=Path,
        default=Path("models/movenet_thunder.tflite"),
    )
    p.add_argument(
        "--lightning",
        type=Path,
        default=Path("models/movenet_lightning.tflite"),
    )
    p.add_argument(
        "--initial",
        choices=["thunder", "lightning"],
        default="thunder",
    )
    p.add_argument("--device", type=int, default=0, help="OpenCV camera device index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Auto-quit after N seconds (0 = run until 'q')",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening a GUI window (smoke / headless)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    estimator = PoseEstimator(
        PoseEstimatorConfig(
            heavy_model_path=args.thunder,
            light_model_path=args.lightning,
            initial_model=args.initial,
        )
    )
    info = estimator.get_model_info()

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

    print(f"runtime: {info['runtime']}")
    print(f"initial model: {estimator.current_model()}")
    print(f"input_shape: {info['input_shape']}  dtype: {info['input_dtype']}")
    window_name = "Edge Inference Guardian (Mac dev)"
    if not args.no_display:
        print("keys: 't' Thunder | 'l' Lightning | 'q' quit")
        print("NOTE: click the OpenCV window first so it has focus")
        # Pre-create the window and pull it to the front (macOS focus help).
        dummy = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        cv2.imshow(window_name, dummy)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
        cv2.waitKey(1)

    fps_ema = 0.0
    last_t = time.perf_counter()
    start_t = last_t
    frames_done = 0
    last_frame_id = 0
    last_pose: PoseResult | None = None
    try:
        while True:
            if args.duration > 0 and (time.perf_counter() - start_t) >= args.duration:
                break

            frame, current_id = cam.read_new_frame(last_frame_id)
            if frame is None:
                if not args.no_display:
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                else:
                    time.sleep(0.005)
                continue
            last_frame_id = current_id

            pose = estimator.estimate(frame)
            last_pose = pose
            frames_done += 1

            now = time.perf_counter()
            inst_fps = 1.0 / max(now - last_t, 1e-6)
            fps_ema = (
                0.9 * fps_ema + 0.1 * inst_fps if fps_ema > 0.0 else inst_fps
            )
            last_t = now

            if not args.no_display:
                overlay = draw_pose(frame, pose.keypoints_array, args.threshold)
                overlay_stats(overlay, pose, fps_ema)
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("t"):
                    estimator.switch_model("thunder")
                if key == ord("l"):
                    estimator.switch_model("lightning")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        elapsed = time.perf_counter() - start_t
        if last_pose is not None:
            print(
                f"finished: elapsed={elapsed:.1f}s  frames={frames_done}  "
                f"last_model={last_pose.model_name}  "
                f"last_inference_ms={last_pose.inference_time_ms:.1f}  "
                f"last_preprocess_ms={last_pose.preprocess_time_ms:.1f}  "
                f"last_avg_conf={last_pose.avg_confidence:.3f}  "
                f"fps_ema={fps_ema:.2f}"
            )
        else:
            print(
                f"finished: elapsed={elapsed:.1f}s  frames=0  "
                f"(no frames processed; check camera permission)"
            )
        print(f"camera stats: {cam.stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
