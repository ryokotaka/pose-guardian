"""Pose estimator wrapping MoveNet Thunder / Lightning.

Spec: 00_機能仕様書.md 3.2 PoseEstimator

Design notes:
- Both Thunder (heavy, 256x256) and Lightning (light, 192x192) are loaded at
  construction time. ``switch_model`` only flips a pointer so model switching
  takes microseconds, not the hundreds of milliseconds of a cold ``allocate_tensors``.
- The price is ~15 MB of RAM (12 MB Thunder + 3 MB Lightning), fine on a 4 GB Pi.
- Runtime is auto-selected: ``tflite_runtime`` → ``ai_edge_litert`` → ``tensorflow.lite``.
- ``estimate`` returns a :class:`PoseResult` containing 17 :class:`Keypoint`
  dataclasses plus measured ``preprocess_time_ms`` and ``inference_time_ms``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ``Interpreter`` import is deferred to :func:`_load_interpreter_class` so this
# module can be imported in environments that have neither tflite_runtime nor
# ai_edge_litert nor tensorflow (e.g. CI without ML deps). Construction of
# :class:`PoseEstimator` is what actually needs a runtime.
RUNTIME: Optional[str] = None


def _load_interpreter_class() -> Tuple[Any, str]:
    """Return ``(Interpreter, runtime_name)`` from the first available runtime.

    Tries ``tflite_runtime`` first (lightest), then ``ai_edge_litert``, then
    full ``tensorflow``. Raises :class:`ImportError` if none are available.
    """
    global RUNTIME
    try:
        from tflite_runtime.interpreter import Interpreter as I  # type: ignore
        RUNTIME = "tflite_runtime"
        return I, RUNTIME
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter as I  # type: ignore
        RUNTIME = "ai_edge_litert"
        return I, RUNTIME
    except ImportError:
        pass
    from tensorflow.lite.python.interpreter import Interpreter as I  # type: ignore
    RUNTIME = "tensorflow.lite"
    return I, RUNTIME


KEYPOINT_NAMES: Tuple[str, ...] = (
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
)


class ModelVariant(str, Enum):
    """String-valued enum so callers can pass either ``"thunder"`` or
    ``ModelVariant.THUNDER`` interchangeably."""

    THUNDER = "thunder"
    LIGHTNING = "lightning"


@dataclass(frozen=True)
class Keypoint:
    """One detected joint."""

    name: str
    y: float        # 0.0-1.0 (image height normalized)
    x: float        # 0.0-1.0 (image width normalized)
    confidence: float


@dataclass
class PoseResult:
    """Result of one inference."""

    keypoints: List[Keypoint]
    model_name: str                       # "thunder" or "lightning"
    inference_time_ms: float
    preprocess_time_ms: float
    input_resolution: Tuple[int, int]     # (height, width)
    avg_confidence: float
    timestamp: float                      # ``time.monotonic()``

    @property
    def keypoints_array(self) -> np.ndarray:
        """Return keypoints as ``(17, 3)`` ndarray of ``[y, x, confidence]``.

        Convenience for code that expects the raw MoveNet output shape
        (e.g. ``examples.sanity_check.draw_pose``).
        """
        return np.array(
            [[kp.y, kp.x, kp.confidence] for kp in self.keypoints],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class PoseEstimatorConfig:
    heavy_model_path: Path = Path("models/movenet_thunder.tflite")
    light_model_path: Path = Path("models/movenet_lightning.tflite")
    initial_model: str = "thunder"        # "thunder" or "lightning"
    num_threads: int = 4


class PoseEstimator:
    """Pose estimator with hot-swappable Thunder/Lightning backends."""

    def __init__(self, config: Optional[PoseEstimatorConfig] = None) -> None:
        self.config = config or PoseEstimatorConfig()
        self._interpreter_class, self._runtime_name = _load_interpreter_class()
        self._interpreters: Dict[str, Any] = {}
        self._load("thunder", Path(self.config.heavy_model_path))
        self._load("lightning", Path(self.config.light_model_path))
        if self.config.initial_model not in self._interpreters:
            raise ValueError(
                f"Unknown initial_model: {self.config.initial_model!r}"
            )
        self._current: str = str(self.config.initial_model)

    # ---- internal ----
    def _load(self, name: str, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}. Run ./models/download_models.sh first."
            )
        Interpreter = self._interpreter_class
        try:
            interp = Interpreter(
                model_path=str(path), num_threads=self.config.num_threads
            )
        except TypeError:
            # Older runtimes (some tflite_runtime versions) don't accept num_threads.
            interp = Interpreter(model_path=str(path))
        interp.allocate_tensors()
        self._interpreters[name] = interp

    @staticmethod
    def _resize_with_pad(image_bgr: np.ndarray, target_size: int) -> np.ndarray:
        """Same algorithm as ``examples.sanity_check.resize_with_pad``.

        Duplicated here to avoid ``src/`` depending on ``examples/``.
        """
        height, width = image_bgr.shape[:2]
        scale = min(target_size / width, target_size / height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        resized = cv2.resize(
            image_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
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

    @classmethod
    def _preprocess(
        cls,
        image_bgr: np.ndarray,
        input_shape: np.ndarray,
        input_dtype: Any,
    ) -> np.ndarray:
        target_height = int(input_shape[1])
        target_width = int(input_shape[2])
        if target_height != target_width:
            resized = cv2.resize(
                image_bgr,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = cls._resize_with_pad(image_bgr, target_height)
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(image_rgb, axis=0)
        if np.issubdtype(input_dtype, np.integer):
            return tensor.astype(input_dtype)
        return tensor.astype(input_dtype) / 255.0

    # ---- public API ----
    def current_model(self) -> str:
        return self._current

    def switch_model(self, name: str) -> float:
        """Switch active model. Idempotent.

        Returns the time spent switching (ms). Near zero because both
        interpreters are already allocated; this just flips a pointer.
        """
        if name not in self._interpreters:
            raise ValueError(f"Unknown model name: {name!r}")
        if name == self._current:
            return 0.0
        t0 = time.perf_counter()
        self._current = name
        return (time.perf_counter() - t0) * 1000.0

    def get_model_info(self) -> dict:
        interp = self._interpreters[self._current]
        inp = interp.get_input_details()[0]
        return {
            "name": self._current,
            "input_shape": tuple(int(x) for x in inp["shape"]),
            "input_dtype": inp["dtype"].__name__,
            "runtime": self._runtime_name,
        }

    def estimate(self, frame_bgr: np.ndarray) -> PoseResult:
        """Run one inference on ``frame_bgr`` (BGR uint8). Returns PoseResult."""
        interp = self._interpreters[self._current]
        input_details = interp.get_input_details()[0]
        output_details = interp.get_output_details()[0]
        h_in = int(input_details["shape"][1])
        w_in = int(input_details["shape"][2])

        t_pre0 = time.perf_counter()
        input_tensor = self._preprocess(
            frame_bgr, input_details["shape"], input_details["dtype"]
        )
        preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0

        t_inf0 = time.perf_counter()
        interp.set_tensor(input_details["index"], input_tensor)
        interp.invoke()
        raw = interp.get_tensor(output_details["index"])[0, 0]  # (17, 3): [y, x, c]
        inference_ms = (time.perf_counter() - t_inf0) * 1000.0

        keypoints = [
            Keypoint(name=name, y=float(y), x=float(x), confidence=float(c))
            for (y, x, c), name in zip(raw, KEYPOINT_NAMES)
        ]
        avg_conf = float(np.mean([k.confidence for k in keypoints]))

        return PoseResult(
            keypoints=keypoints,
            model_name=self._current,
            inference_time_ms=inference_ms,
            preprocess_time_ms=preprocess_ms,
            input_resolution=(h_in, w_in),
            avg_confidence=avg_conf,
            timestamp=time.monotonic(),
        )
