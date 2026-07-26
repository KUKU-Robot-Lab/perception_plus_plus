import numpy as np
import pytest

from perception_plus_plus_core.validation.camera import (
    CameraObservation,
    check_camera_contract,
)


K = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])


def observation(**overrides):
    fields = dict(
        color_encoding="rgb8",
        depth_encoding="16UC1",
        color_shape=(480, 640),
        depth_shape=(480, 640),
        color_frame_id="camera_color_optical_frame",
        depth_frame_id="camera_color_optical_frame",
        info_frame_id="camera_color_optical_frame",
        k=K,
        frames=30,
        duration_s=1.0,
    )
    fields.update(overrides)
    return CameraObservation(**fields)


def status(checks, name):
    return next(check.status for check in checks if check.name == name)


def test_validated_d435i_profile_passes_every_required_check():
    checks = check_camera_contract(observation())
    assert [c.name for c in checks if c.status != "PASS"] == []


def test_unaligned_depth_frame_id_fails():
    checks = check_camera_contract(
        observation(depth_frame_id="camera_depth_optical_frame"))
    assert status(checks, "aligned_optical_frames") == "FAIL"


@pytest.mark.parametrize("shape", [(480, 848), (720, 1280)])
def test_wrong_resolution_profile_fails(shape):
    checks = check_camera_contract(observation(color_shape=shape, depth_shape=shape))
    assert status(checks, "resolution_profile") == "FAIL"


def test_mismatched_image_shapes_fail():
    checks = check_camera_contract(observation(depth_shape=(240, 320)))
    assert status(checks, "image_shapes_match") == "FAIL"


def test_unsupported_encodings_fail():
    checks = check_camera_contract(
        observation(color_encoding="yuv422", depth_encoding="32SC1"))
    assert status(checks, "color_encoding") == "FAIL"
    assert status(checks, "depth_encoding") == "FAIL"


@pytest.mark.parametrize("bad", [
    np.array([[0.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]]),
    np.array([[615.0, 0.0, 0.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]]),
    np.array([[615.0, 0.0, 320.0], [0.0, 615.0, np.nan], [0.0, 0.0, 1.0]]),
])
def test_degenerate_intrinsics_fail(bad):
    checks = check_camera_contract(observation(k=bad))
    assert status(checks, "intrinsics") == "FAIL"


def test_no_delivered_frames_fails():
    checks = check_camera_contract(observation(frames=0, duration_s=2.0))
    assert status(checks, "frame_delivery") == "FAIL"


def test_slow_delivery_is_reported_without_blocking_readiness():
    checks = check_camera_contract(observation(frames=10, duration_s=1.0))
    rate = next(c for c in checks if c.name == "frame_rate")
    assert rate.status == "FAIL"
    assert rate.required is False
