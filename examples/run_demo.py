"""Day 3: Real-time pose estimation on the Mac webcam.

Usage:
    .venv/bin/python examples/run_demo.py
    .venv/bin/python examples/run_demo.py --duration 30
    .venv/bin/python examples/run_demo.py --no-display --duration 5

Keys (when --no-display is NOT set):
    t: switch to Thunder (heavy)
    l: switch to Lightning (light)
    q: quit

Reuses ``examples.sanity_check.prepare_input`` / ``draw_pose`` and
``load_interpreter_class`` for runtime auto-detection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow running this file directly (without installing the package).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.sanity_check import (  # noqa: E402  (sys.path adjustment above)
    draw_pose,
    load_interpreter_class,
    prepare_input,
)
from src.camera import Camera, CameraConfig  # noqa: E402


def load_interpreters(model_paths: dict[str, Path]) -> dict[str, Any]:
    Interpreter = load_interpreter_class()
    interps: dict[str, Any] = {}
    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"model not found: {path}. Run ./models/download_models.sh"
            )
        interp = Interpreter(model_path=str(path))
        interp.allocate_tensors()
        interps[name] = interp
    return interps


def infer_with_timing(interpreter: Any, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Run one inference and return (keypoints[17,3], inference_ms)."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_tensor = prepare_input(
        frame_bgr,
        input_details[0]["shape"],
        input_details[0]["dtype"],
    )

    t0 = time.perf_counter()
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_details[0]["index"])
    inference_ms = (time.perf_counter() - t0) * 1000.0
    return raw[0, 0, :, :], inference_ms


def overlay_stats(
    frame: np.ndarray,
    model_name: str,
    inference_ms: float,
    fps: float,
    avg_conf: float,
) -> None:
    """Draw a black status bar with model / latency / FPS / confidence."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    text = (
        f"{model_name:9s} | {inference_ms:5.1f}ms | "
        f"{fps:4.1f}FPS | conf {avg_conf:.2f}"
    )
    cv2.putText(
        frame,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
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
    interps = load_interpreters({"thunder": args.thunder, "lightning": args.lightning})
    current = args.initial

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

    interp = interps[current]
    runtime_name = f"{type(interp).__module__}.{type(interp).__name__}"
    print(f"runtime: {runtime_name}")
    print(f"initial model: {current}")
    window_name = "Edge Inference Guardian (Mac dev)"
    if not args.no_display:
        print("keys: 't' Thunder | 'l' Lightning | 'q' quit")
        print("NOTE: click the OpenCV window first so it has focus")
        # ダミーフレームでウィンドウを先に作って最前面に出す。
        # macOS だとこれでターミナルから自動でフォーカスが移ることが多い。
        dummy = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        cv2.imshow(window_name, dummy)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass  # 古い OpenCV では未対応。気にしない
        cv2.waitKey(1)

    fps_ema = 0.0
    last_t = time.perf_counter()
    start_t = last_t
    last_inference_ms = 0.0
    last_avg_conf = 0.0
    frames_done = 0
    last_frame_id = 0   # cam.read_new_frame に渡す「最後に処理したフレーム ID」
    try:
        while True:
            if args.duration > 0 and (time.perf_counter() - start_t) >= args.duration:
                break
            # 新しいフレームが届くまで待つ (同じフレームの再処理を避ける)。
            # GUI 動作のため、表示モードでは waitKey も呼んで OS にフォーカス権を返す。
            frame, current_id = cam.read_new_frame(last_frame_id)
            if frame is None:
                if not args.no_display:
                    # ウィンドウを生かしておくため、待ち時間にも waitKey を呼ぶ
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                else:
                    time.sleep(0.005)
                continue
            last_frame_id = current_id

            keypoints, inference_ms = infer_with_timing(interps[current], frame)
            last_inference_ms = inference_ms
            last_avg_conf = float(np.mean(keypoints[:, 2]))
            frames_done += 1

            now = time.perf_counter()
            inst_fps = 1.0 / max(now - last_t, 1e-6)
            fps_ema = (
                0.9 * fps_ema + 0.1 * inst_fps if fps_ema > 0.0 else inst_fps
            )
            last_t = now

            if not args.no_display:
                overlay = draw_pose(frame, keypoints, args.threshold)
                overlay_stats(overlay, current, inference_ms, fps_ema, last_avg_conf)
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("t"):
                    current = "thunder"
                if key == ord("l"):
                    current = "lightning"
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        elapsed = time.perf_counter() - start_t
        print(
            f"finished: elapsed={elapsed:.1f}s  frames={frames_done}  "
            f"last_model={current}  last_inference_ms={last_inference_ms:.1f}  "
            f"last_avg_conf={last_avg_conf:.3f}  fps_ema={fps_ema:.2f}"
        )
        print(f"camera stats: {cam.stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
