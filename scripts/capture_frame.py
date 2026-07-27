#!/usr/bin/env python3
"""Capture one aligned RGB-D frame + intrinsics from a RealSense D435.

Saves a raw NPZ (rgb uint8 HxWx3, depth float32 metres HxW, K 3x3) plus a
preview PNG so the cup framing can be confirmed before running the FP++ smoke.
"""
import argparse

import numpy as np
import pyrealsense2 as rs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/cup_raw.npz")
    ap.add_argument("--preview", default="/tmp/cup_preview.png")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--warmup", type=int, default=30, help="frames to settle AE")
    args = ap.parse_args()

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, 30)
    profile = pipeline.start(cfg)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    try:
        for _ in range(args.warmup):
            pipeline.wait_for_frames()
        frames = align.process(pipeline.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            print("ERROR: no frames")
            return 1

        intr = color.get_profile().as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]], dtype=np.float64)

        bgr = np.asanyarray(color.get_data())
        rgb = bgr[:, :, ::-1].copy()
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale
    finally:
        pipeline.stop()

    np.savez_compressed(args.out, rgb=rgb, depth=depth_m, K=K)
    valid = depth_m[(depth_m > 0.1) & (depth_m < 3.0)]
    print(f"saved {args.out}  rgb={rgb.shape} depth={depth_m.shape} "
          f"K=[fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}]")
    print(f"depth valid[0.1,3.0]m: {valid.size} px, "
          f"median={np.median(valid):.3f}m" if valid.size else "no valid depth")

    try:
        import cv2
        cv2.imwrite(args.preview, bgr)
        print(f"preview {args.preview}")
    except Exception as e:
        print(f"preview skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
