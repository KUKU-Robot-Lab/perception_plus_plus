#!/usr/bin/env python3
"""sim_fpp_batch 시험용 합성 에피소드(입력 계약 형식)를 nvdiffrast 로 렌더한다.

테이블 평면 위에 선 메시를 기울어진 카메라로 본다. 구간:
  static(정지) -> slide(5 mm/프레임 이동 + yaw) -> occl(가림막이 점점 덮음, visible_frac 감소)
  -> gone(물체 화면 밖, LOST 유도) -> back(다른 자리에 다시 나타남, 재초기화 유도)
  -> jump(한 프레임에 큰 이동, POSE_TRANSLATION_JUMP 유도 가능) -> hold
FoundationPose 의 nvdiffrast_render 를 쓰므로 컨테이너(perception-plus-plus 이미지) 안에서 돌린다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("PPP_ROOT", Path(__file__).resolve().parents[2]))
FP_ROOT = REPO / "external/foundationpose_plus_plus"
H, W = 480, 640
K = np.array([[605.0, 0.0, 321.0], [0.0, 605.0, 245.0], [0.0, 0.0, 1.0]])


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """T_world_cam, OpenCV optical(z 앞, y 아래)."""
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


def rot_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(np.eye(3)[np.argmin(np.abs(a))],
                                                                 np.eye(3)[np.argmin(np.abs(a))])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def plane_render(T_world_cam: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """z=0 테이블 평면의 rgb·depth(카메라 z)."""
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    R, p = T_world_cam[:3, :3], T_world_cam[:3, 3]
    d_world = rays @ R.T
    t = -p[2] / np.where(np.abs(d_world[..., 2]) < 1e-9, -1e-9, d_world[..., 2])
    depth = np.where(t > 0, t, 0.0)
    X = p + d_world * depth[..., None]
    checker = ((np.floor(X[..., 0] / 0.05) + np.floor(X[..., 1] / 0.05)) % 2)
    base = 0.55 + 0.12 * checker
    rgb = np.stack([base * 0.95, base * 0.9, base * 0.85], -1)
    rgb = rgb + rng.normal(0, 0.01, rgb.shape)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8), depth.astype(np.float32)


def trajectory(n_seg: int) -> list[tuple[str, np.ndarray, float, float]]:
    """(구간, 테이블 위 xy, yaw, 가림 비율)."""
    out = []
    p0 = np.array([0.0, 0.0])
    for i in range(n_seg):
        out.append(("static", p0.copy(), 0.0, 0.0))
    for i in range(n_seg):
        out.append(("slide", p0 + [0.0, 0.005 * (i + 1)], 0.02 * (i + 1), 0.0))
    p1, y1 = out[-1][1].copy(), out[-1][2]
    for i in range(n_seg):
        out.append(("occl", p1, y1, min(1.0, (i + 1) / n_seg * 0.9)))
    for i in range(n_seg):
        out.append(("gone", None, 0.0, 0.0))
    p2 = np.array([0.06, -0.08])
    for i in range(n_seg + 10):
        out.append(("back", p2, 0.5, 0.0))
    out.append(("jump", p2 + [0.0, 0.30], 0.5, 0.0))
    for i in range(n_seg):
        out.append(("hold", p2 + [0.0, 0.30], 0.5, 0.0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--mesh-scale", type=float, default=1.0)
    ap.add_argument("--sym-axis", default="0,0,1", help="메시 프레임 대칭축(테이블 위로 세울 축)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", type=int, default=20)
    ap.add_argument("--obj-offset", default="0,0,0.03",
                    help="T_obj_mesh 평행이동(물체 원점 -> 메시 원점, m) — 프레임 변환 검사용")
    ap.add_argument("--depth-noise", type=float, default=0.0, help="깊이 가우스 잡음 표준편차 m")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    for p in (FP_ROOT, FP_ROOT / "FoundationPose"):
        sys.path.insert(0, str(p))
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    from FoundationPose.Utils import make_mesh_tensors, nvdiffrast_render

    rng = np.random.default_rng(args.seed)
    mesh = trimesh.load(args.mesh, force="mesh")
    mesh.apply_scale(args.mesh_scale)
    axis = np.array([float(x) for x in args.sym_axis.split(",")])
    axis /= np.linalg.norm(axis)
    tensors = make_mesh_tensors(mesh)
    glctx = dr.RasterizeCudaContext()

    T_world_cam = look_at([-0.55, 0.0, 0.45], [0.0, 0.0, 0.05])
    T_cam_world = np.linalg.inv(T_world_cam)
    R_up = rot_to(axis, np.array([0.0, 0.0, 1.0]))  # 메시 대칭축 -> 테이블 z
    bottom = float((mesh.vertices @ R_up.T)[:, 2].min())
    T_obj_mesh = np.eye(4)
    T_obj_mesh[:3, 3] = [float(x) for x in args.obj_offset.split(",")]

    traj = trajectory(args.seg)
    T = len(traj)
    rgb = np.zeros((T, H, W, 3), np.uint8)
    depth = np.zeros((T, H, W), np.float32)
    mask = np.zeros((T, H, W), np.uint8)
    T_cam_obj = np.full((T, 4, 4), np.nan)
    vis = np.zeros(T, np.float32)
    phases = []
    bg_rgb, bg_depth = plane_render(T_world_cam, rng)
    for t, (phase, xy, yaw, occ) in enumerate(traj):
        phases.append(phase)
        img, dep = bg_rgb.copy(), bg_depth.copy()
        if xy is not None:
            T_world_mesh = np.eye(4)
            T_world_mesh[:3, :3] = rotz(yaw) @ R_up
            T_world_mesh[:3, 3] = [xy[0], xy[1], -bottom]
            T_cam_mesh = T_cam_world @ T_world_mesh
            color, d, _ = nvdiffrast_render(
                K=K, H=H, W=W, ob_in_cams=torch.as_tensor(T_cam_mesh[None], device="cuda",
                                                          dtype=torch.float),
                glctx=glctx, mesh_tensors=tensors, use_light=True)
            color = color[0].cpu().numpy()
            d = d[0].cpu().numpy()
            full = (d > 0) & ((dep == 0) | (d < dep))
            vis_m = full.copy()
            if occ > 0:  # 가림막: 물체 bbox 왼쪽부터 occ 비율을 카메라 앞 0.25 m 판으로 덮음
                ys, xs = np.nonzero(full)
                if xs.size:
                    x_cut = xs.min() + occ * (xs.max() - xs.min() + 1)
                    y0, y1 = max(0, ys.min() - 10), min(H, ys.max() + 10)
                    x0 = max(0, xs.min() - 10)
                    occluder = np.zeros_like(full)
                    occluder[y0:y1, x0:int(x_cut)] = True
                    vis_m = full & ~occluder
            img[full] = np.clip(color[full] * 255, 0, 255).astype(np.uint8)
            dep[full] = d[full]
            if occ > 0:
                img[occluder] = (90, 60, 40)
                dep[occluder] = 0.25
            mask[t] = vis_m.astype(np.uint8)
            vis[t] = vis_m.sum() / max(1, full.sum())
            T_cam_obj[t] = T_cam_mesh @ np.linalg.inv(T_obj_mesh)
        if args.depth_noise > 0:
            dep = np.where(dep > 0, dep + rng.normal(0, args.depth_noise, dep.shape), 0)
        rgb[t], depth[t] = img, dep.astype(np.float32)

    meta = {"object": "synthetic", "fpp_mesh": args.mesh, "fpp_mesh_scale": args.mesh_scale,
            "T_obj_mesh": T_obj_mesh.tolist(), "symmetry_axis": axis.tolist(),
            "phases": phases, "generator": "make_synth_episode.py"}
    T_base_cam = T_world_cam
    T_base_obj = np.einsum("ij,tjk->tik", T_base_cam, T_cam_obj)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, rgb=rgb, depth=depth, K=K, mask_gt=mask, T_cam_obj_gt=T_cam_obj,
                        T_base_cam=T_base_cam, T_base_obj_gt=T_base_obj,
                        tick=np.arange(T), t_s=np.arange(T) / 30.0, visible_frac=vis,
                        meta=json.dumps(meta))
    print(f"wrote {args.out}: {T} frames, phases "
          f"{ {p: phases.count(p) for p in dict.fromkeys(phases)} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
