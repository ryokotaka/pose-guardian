from src.pose_estimator import ModelVariant, PoseEstimator


def test_pose_estimator_keeps_model_metadata() -> None:
    estimator = PoseEstimator("models/movenet_lightning.tflite", ModelVariant.LIGHTNING)

    assert estimator.model_path.name == "movenet_lightning.tflite"
    assert estimator.variant is ModelVariant.LIGHTNING
