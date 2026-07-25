import numpy as np

from perception_plus_plus_core.config import TrackingConfig
from perception_plus_plus_core.testing.fakes import FakeDetector, FakeFpAdapter
from perception_plus_plus_core.tracking.manager import TrackingManager
from perception_plus_plus_core.tracking.state import TrackingState
from perception_plus_plus_core.types import CameraIntrinsics, FrameBundle, MeshSpec, PoseResult


def frame(n):
    return FrameBundle(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32),
                       CameraIntrinsics(1, 1, 1, 1, 4, 4), n, "camera")


def pose(valid=True):
    matrix = np.eye(4)
    matrix[2, 3] = 1.0
    if not valid:
        matrix[0, 0] = np.nan
    return PoseResult(matrix, np.ones((4, 4), bool), 1)


def manager(outputs, detections=None, **config):
    config.setdefault("min_mask_area_px", 1)
    return TrackingManager(
        FakeFpAdapter(outputs),
        FakeDetector(detections or [np.ones((4, 4), bool)]),
        MeshSpec("cup.obj", 1.0),
        TrackingConfig(**config),
    )


def test_initializes_then_tracks_and_only_returns_valid_pose():
    m = manager([pose(), pose()], reinitialize_valid_frames=1)
    first = m.process(frame(1))
    assert first.state is TrackingState.TRACKING
    assert first.pose is not None
    second = m.process(frame(2))
    assert second.pose is not None


def test_invalid_hysteresis_enters_lost_and_resets_adapter():
    adapter = FakeFpAdapter([pose(), pose(False), pose(False)])
    m = TrackingManager(adapter, FakeDetector([np.ones((4, 4), bool)]),
                        MeshSpec("cup.obj", 1),
                        TrackingConfig(max_invalid_frames=2, min_mask_area_px=1,
                                       reinitialize_valid_frames=1))
    m.process(frame(1))
    assert m.process(frame(2)).state is TrackingState.TRACKING
    output = m.process(frame(3))
    assert output.state is TrackingState.LOST
    assert output.pose is None
    assert adapter.reset_count == 1


def test_lost_waits_for_recovery_interval_then_reinitializes():
    detections = [np.ones((4, 4), bool), np.ones((4, 4), bool)]
    m = manager([pose(), pose(False), pose(), pose()], detections=detections,
                max_invalid_frames=1, recovery_interval_frames=2,
                reinitialize_valid_frames=2)
    m.process(frame(1))
    assert m.process(frame(2)).state is TrackingState.LOST
    assert m.process(frame(3)).state is TrackingState.LOST
    assert m.process(frame(4)).state is TrackingState.REINITIALIZING
    assert m.process(frame(5)).state is TrackingState.TRACKING


def test_cuda_oom_is_fatal_and_never_publishes():
    m = manager([MemoryError("CUDA out of memory")])
    output = m.process(frame(1))
    assert output.fatal
    assert output.pose is None
