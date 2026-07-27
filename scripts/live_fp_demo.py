#!/usr/bin/env python3
"""Live FoundationPose++ cup tracking on a RealSense D435, shown in a window.

Pipeline: capture aligned RGB-D -> YOLO cup bbox + depth-band mask (init) ->
FP++ register, then FP++ track() every frame. The RGB view is annotated with
the tracked mask outline, a 3D pose axis, and the cup's camera-frame distance.

Runs inside the perception-plus-plus container. Keys: q=quit, r=re-initialize.

FoundationPose prints dozens of lines per frame from Python AND its C/CUDA
extensions; left alone they throttle the loop to a few fps and tracking loses
fast motion. We silence fd 1 (real stdout) for the whole loop and send our own
status to stderr, so the tracker runs at its true ~15 fps.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, "/workspace/perception_plus_plus")
from perception_plus_plus_core.fp_adapter.foundationpose_plus_plus import (  # noqa: E402
    FoundationPosePlusPlusAdapter,
)
from perception_plus_plus_core.types import (  # noqa: E402
    CameraIntrinsics, FrameBundle, MeshSpec,
)

CUP_CLASS = 41  # COCO 'cup'


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def silence_stdout():
    """Redirect fd 1 to /dev/null (catches FP's Python + C/CUDA chatter)."""
    sys.stdout.flush()
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)


def cup_bbox(model, rgb, conf):
    best = None
    for res in model(rgb, verbose=False):
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
    h, w = depth.shape
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    box = np.zeros((h, w), bool)
    box[y0:y1, x0:x1] = True
    valid = box & (depth > lo) & (depth < hi)
    vals = depth[valid]
    if vals.size == 0:
        return box
    centre = float(np.percentile(vals, 35))
    mask = valid & (depth > centre - band) & (depth < centre + band)
    return mask if mask.sum() >= 2000 else valid


def draw_axis(img, pose, K, length=0.08):
    origin = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).T
    cam = pose[:3, :3] @ origin + pose[:3, 3:4]
    if np.any(cam[2] <= 1e-6):
        return
    u = K[0, 0] * cam[0] / cam[2] + K[0, 2]
    v = K[1, 1] * cam[1] / cam[2] + K[1, 2]
    pts = np.stack([u, v], 1).astype(int)
    o = tuple(pts[0])
    for i, color in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(img, o, tuple(pts[i]), color, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="/workspace/perception_plus_plus/assets/meshes/cup.obj")
    ap.add_argument("--weights", default="/workspace/perception_plus_plus/models/yolo/yolo11n.pt")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO
    yolo = YOLO(args.weights)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    intr_rs = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr_rs.fx, 0, intr_rs.ppx], [0, intr_rs.fy, intr_rs.ppy], [0, 0, 1]])
    intr = CameraIntrinsics(intr_rs.fx, intr_rs.fy, intr_rs.ppx, intr_rs.ppy,
                            args.width, args.height)

    adapter = FoundationPosePlusPlusAdapter("/workspace/perception_plus_plus/external/foundationpose_plus_plus")
    mesh = MeshSpec(args.mesh, 1.0)
    cv2.namedWindow("FoundationPose++ live", cv2.WINDOW_NORMAL)

    log("live demo started (q=quit, r=reinit)")
    silence_stdout()   # from here on FP++ chatter goes to /dev/null

    state = "detect"   # detect -> track
    pose = mask = None
    t_prev, fps, frame_i = time.time(), 0.0, 0

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            cf, df = frames.get_color_frame(), frames.get_depth_frame()
            if not cf or not df:
                continue
            bgr = np.asanyarray(cf.get_data())
            rgb = bgr[:, :, ::-1].copy()
            depth = np.asanyarray(df.get_data()).astype(np.float32) * scale
            view = bgr.copy()

            if state == "detect":
                det = cup_bbox(yolo, rgb, args.conf)
                if det is not None:
                    x0, y0, x1, y1 = [int(v) for v in det[1]]
                    cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    m = bbox_depth_mask(depth, det[1])
                    try:
                        r = adapter.initialize(
                            FrameBundle(rgb, depth, intr, 0, "camera"), m, mesh)
                        pose, mask = r.object_to_camera, r.mask
                        state = "track"
                        log("initialized")
                    except Exception as e:
                        log(f"init failed: {e}")
                cv2.putText(view, "DETECTING cup...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            else:
                r = adapter.track(FrameBundle(rgb, depth, intr, 0, "camera"))
                pose, mask = r.object_to_camera, r.mask
                if mask is not None and mask.any():
                    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(view, cnts, -1, (0, 255, 255), 2)
                draw_axis(view, pose, K)
                z = pose[2, 3]
                cv2.putText(view, f"TRACKING  z={z:.3f}m  {fps:.1f}fps", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            frame_i += 1
            if frame_i % 30 == 0:
                log(f"{state} {fps:.1f}fps"
                    + ("" if pose is None else f" z={pose[2,3]:.3f}m"))

            cv2.imshow("FoundationPose++ live", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                adapter.reset()
                state = "detect"
                log("reinit requested")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
