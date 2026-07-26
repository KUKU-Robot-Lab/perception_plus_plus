import numpy as np
import pytest

from perception_plus_plus_core.detection.yolo import YoloCupDetector
from perception_plus_plus_core.errors import ModelLoadError
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


def test_yolo_rejects_detection_output_without_instance_masks():
    class DetectionResult:
        boxes, masks = Boxes(), None

    detector = YoloCupDetector(
        "unused.pt", 41, 0.5, model=lambda _: [DetectionResult()])
    with pytest.raises(ModelLoadError, match="segmentation"):
        detector.detect(np.zeros((2, 2, 3), np.uint8))


def test_yolo_returns_nothing_for_an_empty_frame_without_faulting_the_model():
    # Ultralytics reports masks=None together with an empty Boxes when a
    # segmentation checkpoint finds no instance; that is not a load error.
    class EmptyBoxes:
        cls = np.array([])
        conf = np.array([])
        xyxy = np.zeros((0, 4))

        def __len__(self):
            return 0

    class EmptyResult:
        boxes, masks = EmptyBoxes(), None

    detector = YoloCupDetector("unused.pt", 41, 0.5, model=lambda _: [EmptyResult()])
    assert detector.detect(np.zeros((2, 2, 3), np.uint8)) == []


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


def test_fp_adapter_links_cutie_models_to_upstream_expected_directory(tmp_path):
    root = tmp_path / "upstream"
    (root / "Cutie").mkdir(parents=True)
    model_root = tmp_path / "models"
    source = model_root / "cutie/cutie-base-mega.pth"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"cutie")
    from perception_plus_plus_core.fp_adapter.foundationpose_plus_plus import _UpstreamEngine

    _UpstreamEngine._link_models(root, model_root)
    target = root / "Cutie/weights/cutie-base-mega.pth"
    assert target.is_symlink()
    assert target.resolve() == source.resolve()
