import numpy as np

from examples.sanity_check import draw_pose, prepare_input, resize_with_pad


def test_resize_with_pad_outputs_square_image() -> None:
    image = np.zeros((40, 80, 3), dtype=np.uint8)

    output = resize_with_pad(image, 64)

    assert output.shape == (64, 64, 3)


def test_prepare_input_matches_model_shape_and_dtype() -> None:
    image = np.zeros((40, 80, 3), dtype=np.uint8)

    input_tensor = prepare_input(image, np.array([1, 192, 192, 3]), np.uint8)

    assert input_tensor.shape == (1, 192, 192, 3)
    assert input_tensor.dtype == np.uint8


def test_draw_pose_keeps_image_shape() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 0] = 0.5
    keypoints[:, 1] = 0.5
    keypoints[:, 2] = 0.9

    output = draw_pose(image, keypoints, threshold=0.3)

    assert output.shape == image.shape
