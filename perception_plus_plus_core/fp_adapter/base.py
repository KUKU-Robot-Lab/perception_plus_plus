from typing import Protocol
import numpy as np

from ..types import FrameBundle, MeshSpec, PoseResult


class FpAdapter(Protocol):
    def initialize(self, frame: FrameBundle, mask: np.ndarray, mesh: MeshSpec) -> PoseResult: ...
    def track(self, frame: FrameBundle) -> PoseResult: ...
    def reset(self) -> None: ...

