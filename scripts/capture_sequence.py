#!/usr/bin/env python3
"""Capture an aligned RGB-D sequence + a frame-0 cup mask (YOLO bbox + depth).

Writes seq npz: rgb[N,H,W,3] uint8, depth[N,H,W] float32 m, K 3x3, mask0 HxW.
Host-side (RealSense + modern ultralytics live in the .venv).
"""
import argparse
import numpy as np
import pyrealsense2 as rs

CUP_CLASS = 41


def cup_bbox(rgb, weights, conf):
    from ultralytics import YOLO
    best = None
    for res in YOLO(weights)(rgb, verbose=False):
        if res.boxes is None:
            continue
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cfd = res.boxes.conf.cpu().numpy()
        box = res.boxes.xyxy.cpu().numpy()
        for i, (c, p) in enumerate(zip(cls, cfd)):
            if c == CUP_CLASS and p >= conf and (best is None or p > best[0]):
                best = (float(p), box[i])
    return best


def bbox_depth_mask(depth, xyxy, band=0.08, lo=0.1, hi=3.0):
    # cup occupies the nearer depths inside the box; centre the band on a low
    # percentile (robust to background seen around the cup) and keep a wide
    # enough band to span the cup body. Fall back to all valid-depth-in-box.
    h, w = depth.shape
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    box = np.zeros((h, w), bool); box[y0:y1, x0:x1] = True
    valid_in_box = box & (depth > lo) & (depth < hi)
    vals = depth[valid_in_box]
    if vals.size == 0:
        return box
    centre = float(np.percentile(vals, 35))
    mask = valid_in_box & (depth > centre - band) & (depth < centre + band)
    if mask.sum() < 2000:              # band missed the body -> use all valid
        mask = valid_in_box
    return mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ppp_io/cup_seq.npz")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--weights", default="models/yolo/yolo11n.pt")
    args = ap.parse_args()

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    rgbs, depths = [], []
    K = None
    try:
        for _ in range(args.warmup):
            pipeline.wait_for_frames()
        for _ in range(args.n):
            fr = align.process(pipeline.wait_for_frames())
            c, d = fr.get_color_frame(), fr.get_depth_frame()
            if not c or not d:
                continue
            if K is None:
                intr = c.get_profile().as_video_stream_profile().get_intrinsics()
                K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy],
                              [0, 0, 1]], np.float64)
            rgbs.append(np.asanyarray(c.get_data())[:, :, ::-1].copy())
            depths.append(np.asanyarray(d.get_data()).astype(np.float32) * scale)
    finally:
        pipeline.stop()

    rgb = np.stack(rgbs); depth = np.stack(depths)
    det = cup_bbox(rgb[0], args.weights, 0.25)
    if det is None:
        print("no cup in frame 0"); return 1
    mask0 = bbox_depth_mask(depth[0], det[1])
    np.savez_compressed(args.out, rgb=rgb, depth=depth, K=K, mask0=mask0.astype(bool))
    print(f"saved {args.out}  N={len(rgb)} rgb={rgb.shape} "
          f"cup_conf={det[0]:.3f} mask0={int(mask0.sum())}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
