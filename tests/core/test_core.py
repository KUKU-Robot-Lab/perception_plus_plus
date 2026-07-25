import numpy as np
import pytest

from perception_plus_plus_core.config import TrackingConfig
from perception_plus_plus_core.types import CameraIntrinsics, FrameBundle, PoseResult
from perception_plus_plus_core.validation.depth import depth_to_meters
from perception_plus_plus_core.validation.geometry import pose_delta
from perception_plus_plus_core.validation.quality import evaluate_quality


def frame(depth=None):
    return FrameBundle(
        rgb=np.zeros((4, 5, 3), np.uint8),
        depth=np.ones((4, 5), np.float32) if depth is None else depth,
        intrinsics=CameraIntrinsics(100, 100, 2, 2, 5, 4),
        timestamp_ns=1,
        frame_id="camera_color_optical_frame",
    )


def result(x=0.0, mask=True):
    pose = np.eye(4)
    pose[0, 3] = x
    pose[2, 3] = 1.0
    return PoseResult(pose, np.ones((4, 5), bool) if mask else None, 1)


def test_frame_rejects_mismatched_depth():
    with pytest.raises(ValueError, match="shape"):
        frame(np.ones((3, 5), np.float32))


def test_depth_to_meters_supports_uint16_and_rejects_encoding():
    assert depth_to_meters(np.array([[1000]], np.uint16), "16UC1")[0, 0] == 1.0
    with pytest.raises(ValueError, match="encoding"):
        depth_to_meters(np.ones((1, 1)), "8UC1")


def test_pose_delta_returns_translation_and_rotation():
    a, b = np.eye(4), np.eye(4)
    b[0, 3] = 0.2
    b[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    translation, rotation = pose_delta(a, b)
    assert translation == pytest.approx(0.2)
    assert rotation == pytest.approx(90.0)


def test_quality_rejects_jump_and_accepts_valid_pose():
    cfg = TrackingConfig(min_mask_area_px=3, max_translation_m=0.1)
    assert evaluate_quality(frame(), result(), None, cfg).valid
    decision = evaluate_quality(frame(), result(0.2), result(), cfg)
    assert not decision.valid
    assert decision.reason == "POSE_TRANSLATION_JUMP"
