"""Run one MoveNet TFLite model on one image and save a keypoint overlay.

Usage:
    .venv/bin/python examples/sanity_check.py tmp/test.jpg models/movenet_thunder.tflite
    .venv/bin/python examples/sanity_check.py tmp/test.jpg models/movenet_lightning.tflite \
        --output tmp/sanity_check_output_lightning.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def load_interpreter_class() -> type[Any]:
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter
    except ImportError:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except ImportError:
        pass

    from tensorflow.lite.python.interpreter import Interpreter

    return Interpreter


def resize_with_pad(image_bgr: np.ndarray, target_size: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scale = min(target_size / width, target_size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    top = (target_size - resized_height) // 2
    bottom = target_size - resized_height - top
    left = (target_size - resized_width) // 2
    right = target_size - resized_width - left

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def prepare_input(image_bgr: np.ndarray, input_shape: np.ndarray, input_dtype: np.dtype) -> np.ndarray:
    target_height = int(input_shape[1])
    target_width = int(input_shape[2])
    if target_height != target_width:
        resized = cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        resized = resize_with_pad(image_bgr, target_height)

    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(image_rgb, axis=0)

    if np.issubdtype(input_dtype, np.integer):
        return input_tensor.astype(input_dtype)
    return input_tensor.astype(input_dtype) / 255.0


def draw_pose(image_bgr: np.ndarray, keypoints: np.ndarray, threshold: float) -> np.ndarray:
    output = image_bgr.copy()
    height, width = output.shape[:2]

    for start, end in EDGES:
        y1, x1, c1 = keypoints[start]
        y2, x2, c2 = keypoints[end]
        if c1 < threshold or c2 < threshold:
            continue
        p1 = (int(x1 * width), int(y1 * height))
        p2 = (int(x2 * width), int(y2 * height))
        cv2.line(output, p1, p2, (255, 180, 0), 2)

    for index, (y, x, confidence) in enumerate(keypoints):
        if confidence < threshold:
            continue
        center = (int(x * width), int(y * height))
        cv2.circle(output, center, 4, (0, 255, 0), -1)
        cv2.putText(
            output,
            KEYPOINT_NAMES[index],
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def run_inference(image_path: Path, model_path: Path, output_path: Path, threshold: float) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    Interpreter = load_interpreter_class()
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]
    input_tensor = prepare_input(image_bgr, input_shape, input_dtype)

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    raw_keypoints = interpreter.get_tensor(output_details[0]["index"])
    keypoints = raw_keypoints[0, 0, :, :]

    target_size = int(input_shape[1])
    display_image = resize_with_pad(image_bgr, target_size)
    overlay = draw_pose(display_image, keypoints, threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)

    print(f"runtime: {Interpreter.__module__}.{Interpreter.__name__}")
    print(f"model: {model_path}")
    print(f"input_shape: {tuple(input_shape)}")
    print(f"input_dtype: {input_dtype}")
    print(f"output_shape: {tuple(raw_keypoints.shape)}")
    print(f"mean_confidence: {float(np.mean(keypoints[:, 2])):.4f}")
    print(f"max_confidence: {float(np.max(keypoints[:, 2])):.4f}")
    print(f"wrote: {output_path}")
    return keypoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-image MoveNet sanity check.")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("model", type=Path, help="MoveNet .tflite model path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/sanity_check_output.jpg"),
        help="Output overlay image path",
    )
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum keypoint confidence to draw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(args.image, args.model, args.output, args.threshold)


if __name__ == "__main__":
    main()
