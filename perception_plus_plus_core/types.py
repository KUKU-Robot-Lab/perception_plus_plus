from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
                        dtype=np.float64)


@dataclass(frozen=True)
class FrameBundle:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp_ns: int
    frame_id: str

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb shape must be HxWx3")
        if self.depth.shape != self.rgb.shape[:2]:
            raise ValueError("depth shape must match rgb")
        if self.rgb.shape[:2] != (self.intrinsics.height, self.intrinsics.width):
            raise ValueError("intrinsics dimensions must match image shape")
        if not self.frame_id:
            raise ValueError("frame_id cannot be empty")


@dataclass(frozen=True)
class MeshSpec:
    path: str | Path
    scale_to_meters: float

    def __post_init__(self) -> None:
        if self.scale_to_meters <= 0:
            raise ValueError("mesh scale must be positive")


@dataclass(frozen=True)
class PoseResult:
    object_to_camera: np.ndarray
    mask: np.ndarray | None
    timestamp_ns: int
    score: float | None = None
    diagnostic: str = ""

    def __post_init__(self) -> None:
        if np.shape(self.object_to_camera) != (4, 4):
            raise ValueError("pose shape must be 4x4")
        if self.mask is not None and self.mask.ndim != 2:
            raise ValueError("mask shape must be HxW")

