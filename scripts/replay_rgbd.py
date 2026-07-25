#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception_plus_plus_core.config import TrackingConfig
from perception_plus_plus_core.detection.yolo import YoloCupDetector
from perception_plus_plus_core.fp_adapter.foundationpose_plus_plus import FoundationPosePlusPlusAdapter
from perception_plus_plus_core.tracking.manager import TrackingManager
from perception_plus_plus_core.types import MeshSpec
from perception_plus_plus_core.validation.replay import ReplayReader, run_replay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--class-id", type=int, default=41)
    parser.add_argument("--config", type=Path, default=Path("config/cup_tracking.yaml"))
    args = parser.parse_args()
    manager = TrackingManager(
        FoundationPosePlusPlusAdapter(),
        YoloCupDetector(args.yolo, args.class_id, 0.5),
        MeshSpec(args.mesh, args.mesh_scale),
        TrackingConfig.from_yaml(args.config))
    print(run_replay(manager, ReplayReader(args.recording, "camera_color_optical_frame")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
