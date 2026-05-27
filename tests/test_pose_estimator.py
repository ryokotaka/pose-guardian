"""PoseEstimator smoke tests.

These need the real MoveNet tflite files. If models are not present,
all tests in this module are skipped so the suite stays green on CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


THUNDER = Path("models/movenet_thunder.tflite")
LIGHTNING = Path("models/movenet_lightning.tflite")


@pytest.fixture(scope="module")
def estimator():
    if not (THUNDER.exists() and LIGHTNING.exists()):
        pytest.skip(
            "MoveNet tflite models not present; run ./models/download_models.sh"
        )
    from src.pose_estimator import PoseEstimator, PoseEstimatorConfig

    try:
        return PoseEstimator(PoseEstimatorConfig())
    except ImportError as exc:
        # No TFLite runtime available (e.g. plain CI without ai-edge-litert).
        pytest.skip(f"No TFLite runtime available: {exc}")


def test_estimate_returns_seventeen_keypoints(estimator):
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    result = estimator.estimate(dummy)

    assert len(result.keypoints) == 17
    assert result.model_name == "thunder"
    assert result.inference_time_ms > 0
    assert result.preprocess_time_ms >= 0
    assert result.input_resolution == (256, 256)


def test_keypoint_field_ranges(estimator):
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    result = estimator.estimate(dummy)

    for kp in result.keypoints:
        assert 0.0 <= kp.confidence <= 1.0
        assert isinstance(kp.name, str)


def test_keypoints_array_shape(estimator):
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    result = estimator.estimate(dummy)

    arr = result.keypoints_array
    assert arr.shape == (17, 3)
    assert arr.dtype == np.float32


def test_switch_model_same_returns_zero(estimator):
    estimator.switch_model("thunder")

    delta = estimator.switch_model("thunder")

    assert delta == 0.0


def test_switch_model_round_trip(estimator):
    estimator.switch_model("lightning")
    assert estimator.current_model() == "lightning"

    estimator.switch_model("thunder")
    assert estimator.current_model() == "thunder"


def test_switch_model_unknown_raises(estimator):
    with pytest.raises(ValueError):
        estimator.switch_model("yolo")


def test_get_model_info_thunder(estimator):
    estimator.switch_model("thunder")

    info = estimator.get_model_info()

    assert info["name"] == "thunder"
    assert info["input_shape"] == (1, 256, 256, 3)
    assert "runtime" in info


def test_get_model_info_lightning(estimator):
    estimator.switch_model("lightning")
    try:
        info = estimator.get_model_info()
        assert info["name"] == "lightning"
        assert info["input_shape"] == (1, 192, 192, 3)
    finally:
        estimator.switch_model("thunder")  # leave the fixture in a known state


def test_inference_actually_runs(estimator):
    """A non-zero image should yield a result without raising."""
    estimator.switch_model("lightning")
    try:
        # Random noise; we don't care about correctness, only that the pipeline runs.
        rng = np.random.default_rng(seed=0)
        frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

        result = estimator.estimate(frame)

        assert 0.0 <= result.avg_confidence <= 1.0
        assert result.inference_time_ms > 0
    finally:
        estimator.switch_model("thunder")


def test_config_defaults_match_spec():
    """Verify the dataclass defaults so callers can rely on them."""
    from src.pose_estimator import PoseEstimatorConfig

    cfg = PoseEstimatorConfig()
    assert str(cfg.heavy_model_path) == "models/movenet_thunder.tflite"
    assert str(cfg.light_model_path) == "models/movenet_lightning.tflite"
    assert cfg.initial_model == "thunder"
    assert cfg.num_threads == 4


def test_model_variant_enum_values():
    """ModelVariant should be a str-valued enum that compares to strings."""
    from src.pose_estimator import ModelVariant

    assert ModelVariant.THUNDER == "thunder"
    assert ModelVariant.LIGHTNING == "lightning"
