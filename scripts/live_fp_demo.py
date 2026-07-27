#!/usr/bin/env python3
"""Live FoundationPose++ cup tracking on a RealSense D435, shown in a window.

Detection-anchored tracking. Every frame YOLO detects the cup box (COCO class
41) and FP++ tracks the 6D pose. If the tracked pose reprojects OUTSIDE the
detected cup box for a few frames (it has drifted onto the hand while lifting),
or leaves a plausible depth range, the tracker RE-ANCHORS on the cup from a
fresh bbox+depth mask. This mirrors FP++'s "CAD + bbox" behaviour.

Re-anchor is done in place (estimator.register + cutie.initialize on the
existing engine): FP++/Cutie cannot be fully re-initialized twice in one
process (Hydra + CUDA global state), so adapter.initialize() is called exactly
once and every later recovery reuses that engine.

Runs inside the perception-plus-plus container. Keys: q=quit, r=re-anchor.
FoundationPose prints per frame from Python and C/CUDA; we silence fd 1 for the
whole loop and send status to stderr.
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

CUP_CLASS = 41


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def silence_stdout():
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


def as_mat(pose):
    p = pose.detach().cpu().numpy() if hasattr(pose, "detach") else np.asarray(pose)
    return p.reshape(4, 4)


def reanchor(adapter, rgb, depth, Km, mask, mask_u8):
    """Re-estimate pose + reset the Cutie mask on the existing engine."""
    eng = adapter.engine
    pose = eng.estimator.register(K=Km, rgb=rgb, depth=depth,
                                  ob_mask=mask_u8 * 255, iteration=eng.est_iter)
    eng.cutie.initialize(rgb, {"mask": mask_u8})
    eng.mask = mask
    eng.kf_mean, eng.kf_covariance = eng.kalman.initiate(eng.get_pose_array(pose))
    return as_mat(pose), mask


def reproject(pose, K):
    p = pose[:3, 3]
    if p[2] <= 1e-6:
        return None
    return (K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2])


def inside_box(uv, xyxy, margin):
    x0, y0, x1, y1 = xyxy
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return (x0 - mx <= uv[0] <= x1 + mx) and (y0 - my <= uv[1] <= y1 + my)


def draw_axis(img, pose, K, length=0.08):
    o3 = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).T
    cam = pose[:3, :3] @ o3 + pose[:3, 3:4]
    if np.any(cam[2] <= 1e-6):
        return
    u = K[0, 0] * cam[0] / cam[2] + K[0, 2]
    v = K[1, 1] * cam[1] / cam[2] + K[1, 2]
    pts = np.stack([u, v], 1).astype(int)
    o = tuple(pts[0])
    for i, c in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(img, o, tuple(pts[i]), c, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="/workspace/perception_plus_plus/assets/meshes/cup.obj")
    ap.add_argument("--weights", default="/workspace/perception_plus_plus/models/yolo/yolo11n.pt")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.35, help="bbox anchor tolerance")
    ap.add_argument("--z-min", type=float, default=0.15)
    ap.add_argument("--z-max", type=float, default=1.6)
    ap.add_argument("--patience", type=int, default=3, help="bad frames before re-anchor")
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
    ir = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[ir.fx, 0, ir.ppx], [0, ir.fy, ir.ppy], [0, 0, 1]])
    intr = CameraIntrinsics(ir.fx, ir.fy, ir.ppx, ir.ppy, args.width, args.height)

    adapter = FoundationPosePlusPlusAdapter("/workspace/perception_plus_plus/external/foundationpose_plus_plus")
    mesh = MeshSpec(args.mesh, 1.0)
    cv2.namedWindow("FoundationPose++ live", cv2.WINDOW_NORMAL)

    log("live demo started (q=quit, r=re-anchor)")
    silence_stdout()

    started = False        # first adapter.initialize done?
    pose = mask = None
    bad = 0
    fps, t_prev, fi = 0.0, time.time(), 0

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

            det = cup_bbox(yolo, rgb, args.conf)
            if det is not None:
                x0, y0, x1, y1 = [int(v) for v in det[1]]
                cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)

            status, color = "", (0, 255, 0)
            if not started:
                if det is not None:
                    m = bbox_depth_mask(depth, det[1])
                    try:
                        r = adapter.initialize(
                            FrameBundle(rgb, depth, intr, 0, "camera"), m, mesh)
                        pose, mask = r.object_to_camera, r.mask
                        started, bad = True, 0
                        log("initialized")
                    except Exception as e:
                        log(f"init failed: {e}")
                status, color = "DETECTING cup...", (0, 200, 255)
            else:
                r = adapter.track(FrameBundle(rgb, depth, intr, 0, "camera"))
                pose, mask = r.object_to_camera, r.mask
                uv = reproject(pose, K)
                z = pose[2, 3]
                off_cup = det is not None and uv is not None and not inside_box(uv, det[1], args.margin)
                invalid = (z < args.z_min or z > args.z_max) or off_cup
                if invalid:
                    bad += 1
                    if det is not None and bad >= args.patience:
                        m = bbox_depth_mask(depth, det[1])
                        try:
                            pose, mask = reanchor(adapter, rgb, depth, K, m,
                                                  m.astype(np.uint8))
                            bad = 0
                            log("re-anchored")
                        except Exception as e:
                            log(f"reanchor failed: {e}")
                        status, color = "RE-ANCHOR", (0, 165, 255)
                    else:
                        status, color = "DRIFT? (searching cup)", (0, 0, 255)
                else:
                    bad = 0
                    status, color = f"TRACKING z={z:.3f}m", (0, 255, 0)

                if mask is not None and mask.any():
                    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(view, cnts, -1, (0, 255, 255), 2)
                draw_axis(view, pose, K)

            cv2.putText(view, f"{status}  {fps:.1f}fps", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            fi += 1
            if fi % 30 == 0:
                log(f"{'track' if started else 'detect'} {fps:.1f}fps bad={bad}")

            cv2.imshow("FoundationPose++ live", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                bad = args.patience     # force re-anchor on next detected cup
                log("re-anchor requested")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
