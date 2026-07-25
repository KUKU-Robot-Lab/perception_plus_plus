from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np

from ..types import CameraIntrinsics, FrameBundle


class ReplayReader:
    def __init__(self, path: str | Path, frame_id: str) -> None:
        data = np.load(path, allow_pickle=False)
        self.rgb, self.depth = data["rgb"], data["depth"]
        self.timestamps = data["timestamps_ns"]
        k = data["K"]
        if len(self.rgb) != len(self.depth) or len(self.rgb) != len(self.timestamps):
            raise ValueError("replay arrays must have the same frame count")
        height, width = self.rgb.shape[1:3]
        self.intrinsics = CameraIntrinsics(float(k[0, 0]), float(k[1, 1]),
                                           float(k[0, 2]), float(k[1, 2]), width, height)
        self.frame_id = frame_id

    def __iter__(self):
        for index in np.argsort(self.timestamps):
            yield FrameBundle(self.rgb[index], self.depth[index], self.intrinsics,
                              int(self.timestamps[index]), self.frame_id)


@dataclass(frozen=True)
class ReplayReport:
    frames: int
    publishable_poses: int
    mean_latency_ms: float
    states: dict[str, int]


def run_replay(manager, reader) -> ReplayReport:
    latencies, publishable, states = [], 0, {}
    for frame in reader:
        start = perf_counter()
        output = manager.process(frame)
        latencies.append((perf_counter() - start) * 1000)
        publishable += output.pose is not None
        states[output.state.name] = states.get(output.state.name, 0) + 1
    return ReplayReport(len(latencies), publishable,
                        float(np.mean(latencies)) if latencies else 0.0, states)

