"""Camera logic tests that do not need a real webcam."""

import numpy as np
import pytest

from src.camera import Camera, CameraConfig


def test_default_config_values() -> None:
    config = CameraConfig()

    assert config.source == "opencv"
    assert config.device_index == 0
    assert config.width == 640
    assert config.height == 480
    assert config.fps_cap == 30


def test_camera_not_alive_before_start() -> None:
    cam = Camera(CameraConfig())

    assert cam.is_alive() is False
    assert cam.read_frame() is None


def test_camera_stats_initial_values() -> None:
    cam = Camera(CameraConfig())

    stats = cam.stats()

    assert stats["frames_read"] == 0
    assert stats["frames_dropped"] == 0
    assert stats["frame_id"] == 0
    assert stats["is_alive"] is False


def test_unsupported_source_raises() -> None:
    cam = Camera(CameraConfig(source="picamera"))

    with pytest.raises(NotImplementedError):
        cam.start()


def test_stop_is_idempotent_when_not_started() -> None:
    cam = Camera(CameraConfig())

    cam.stop()
    cam.stop()  # 2回呼んでも壊れない

    assert cam.is_alive() is False


# ---- read_new_frame ----

def test_read_new_frame_returns_none_before_any_frame() -> None:
    cam = Camera(CameraConfig())

    frame, frame_id = cam.read_new_frame(last_id=0)

    assert frame is None
    assert frame_id == 0


def test_read_new_frame_after_capture_returns_new_frame() -> None:
    """capture スレッドを起こさずに直接 _latest を set してロジックを検証する。"""
    cam = Camera(CameraConfig())
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    # 内部状態を直接更新 (capture スレッドの動作をシミュレート)
    with cam._lock:
        cam._latest = dummy
        cam._frame_id = 1

    frame, current_id = cam.read_new_frame(last_id=0)

    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert current_id == 1


def test_read_new_frame_returns_none_when_id_unchanged() -> None:
    """同じ frame_id を渡すと「新しいフレーム無し」を返す。"""
    cam = Camera(CameraConfig())
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    with cam._lock:
        cam._latest = dummy
        cam._frame_id = 5

    frame, current_id = cam.read_new_frame(last_id=5)

    assert frame is None
    assert current_id == 5


def test_read_new_frame_returns_copy_not_reference() -> None:
    """書き換えても内部 _latest が壊れないこと。"""
    cam = Camera(CameraConfig())
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    with cam._lock:
        cam._latest = dummy
        cam._frame_id = 1

    frame, _ = cam.read_new_frame(last_id=0)
    assert frame is not None
    frame[0, 0, 0] = 255

    assert cam._latest[0, 0, 0] == 0  # 元の中身は変わっていない
