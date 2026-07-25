#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception_plus_plus_core.fp_adapter.foundationpose_plus_plus import (
    FoundationPosePlusPlusAdapter,
)
from perception_plus_plus_core.types import CameraIntrinsics, FrameBundle, MeshSpec


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one FP++ initialization frame")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--upstream", type=Path,
                        default=Path("external/foundationpose_plus_plus"))
    args = parser.parse_args()
    data = np.load(args.npz, allow_pickle=False)
    rgb, depth, mask, k = data["rgb"], data["depth"], data["mask"], data["K"]
    if rgb.ndim == 4:
        rgb, depth, mask = rgb[0], depth[0], mask[0]
    height, width = rgb.shape[:2]
    frame = FrameBundle(rgb, depth.astype(np.float32),
                        CameraIntrinsics(k[0, 0], k[1, 1], k[0, 2], k[1, 2],
                                         width, height), 0, "camera")
    result = FoundationPosePlusPlusAdapter(args.upstream).initialize(
        frame, mask.astype(bool), MeshSpec(args.mesh, args.mesh_scale))
    print(np.array2string(result.object_to_camera, precision=6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
