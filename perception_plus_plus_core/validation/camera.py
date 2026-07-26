from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .readiness import CheckResult

COLOR_ENCODINGS = {"rgb8", "bgr8"}
DEPTH_ENCODINGS = {"16UC1", "mono16", "32FC1"}


@dataclass(frozen=True)
class CameraObservation:
    """One synchronized RGB-D delivery described without any ROS types."""

    color_encoding: str
    depth_encoding: str
    color_shape: tuple[int, int]
    depth_shape: tuple[int, int]
    color_frame_id: str
    depth_frame_id: str
    info_frame_id: str
    k: np.ndarray
    frames: int
    duration_s: float


def check_camera_contract(
    observation: CameraObservation,
    expected_width: int = 640,
    expected_height: int = 480,
    expected_hz: float = 30.0,
    minimum_hz_ratio: float = 0.8,
) -> list[CheckResult]:
    """Validate the D435i profile the tracker depends on.

    The tracker consumes aligned, synchronized RGB-D, so the depth image must
    carry the colour optical frame and both images must share one size.
    """
    checks: list[CheckResult] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append(CheckResult(name, "PASS" if ok else "FAIL", required, detail))

    add("color_encoding", observation.color_encoding in COLOR_ENCODINGS,
        observation.color_encoding)
    add("depth_encoding", observation.depth_encoding in DEPTH_ENCODINGS,
        observation.depth_encoding)
    add("image_shapes_match", observation.color_shape == observation.depth_shape,
        f"color={observation.color_shape} depth={observation.depth_shape}")
    add("resolution_profile",
        observation.color_shape == (expected_height, expected_width),
        f"{observation.color_shape[1]}x{observation.color_shape[0]} "
        f"expected {expected_width}x{expected_height}")
    aligned = (observation.depth_frame_id == observation.color_frame_id
               and observation.info_frame_id == observation.color_frame_id
               and bool(observation.color_frame_id))
    add("aligned_optical_frames", aligned,
        f"color={observation.color_frame_id} depth={observation.depth_frame_id} "
        f"info={observation.info_frame_id}")

    k = np.asarray(observation.k, dtype=np.float64).reshape(3, 3)
    height, width = observation.color_shape
    intrinsics_ok = bool(
        np.all(np.isfinite(k))
        and k[0, 0] > 0 and k[1, 1] > 0
        and 0 < k[0, 2] < width and 0 < k[1, 2] < height)
    add("intrinsics", intrinsics_ok,
        f"fx={k[0, 0]:.3f} fy={k[1, 1]:.3f} cx={k[0, 2]:.3f} cy={k[1, 2]:.3f}")

    add("frame_delivery", observation.frames > 0, f"{observation.frames} synchronized frames")
    rate = observation.frames / observation.duration_s if observation.duration_s > 0 else 0.0
    add("frame_rate", rate >= expected_hz * minimum_hz_ratio,
        f"{rate:.2f} Hz over {observation.duration_s:.2f}s "
        f"(minimum {expected_hz * minimum_hz_ratio:.2f} Hz)",
        required=False)
    return checks
