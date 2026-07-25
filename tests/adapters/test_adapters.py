import numpy as np

from perception_plus_plus_core.detection.yolo import YoloCupDetector
from perception_plus_plus_core.fp_adapter.foundationpose_plus_plus import FoundationPosePlusPlusAdapter
from perception_plus_plus_core.types import CameraIntrinsics, FrameBundle, MeshSpec


class Boxes:
    cls = np.array([41, 1])
    conf = np.array([0.9, 0.8])
    xyxy = np.array([[0, 0, 2, 2], [0, 0, 1, 1]])


class Masks:
    data = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 1]]])


class Result:
    boxes, masks = Boxes(), Masks()


def test_yolo_normalizes_and_filters_results():
    detector = YoloCupDetector("unused.pt", 41, 0.5, model=lambda _: [Result()])
    detections = detector.detect(np.zeros((2, 2, 3), np.uint8))
    assert len(detections) == 1
    assert detections[0].mask.dtype == bool


def test_fp_adapter_delegates_frame_api_and_reset():
    calls = []

    class Engine:
        def initialize(self, rgb, depth, mask, intrinsics, mesh):
            calls.append("initialize")
            return np.eye(4), mask

        def track(self, rgb, depth, intrinsics):
            calls.append("track")
            return np.eye(4), np.ones((2, 2), bool)

        def reset(self):
            calls.append("reset")

    frame = FrameBundle(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.float32),
                        CameraIntrinsics(1, 1, 1, 1, 2, 2), 7, "camera")
    adapter = FoundationPosePlusPlusAdapter(engine=Engine())
    assert adapter.initialize(frame, np.ones((2, 2), bool), MeshSpec("cup.obj", 1)).timestamp_ns == 7
    assert adapter.track(frame).mask.shape == (2, 2)
    adapter.reset()
    assert calls == ["initialize", "track", "reset"]

