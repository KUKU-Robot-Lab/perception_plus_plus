#!/usr/bin/env python3
"""서베이 폴더들의 프레임에서 테이블 위 컵의 base 좌표를 구한다(z 고정 + CAD 실루엣 맞춤, 여러 자세 통합).
    python3 scripts/estimate_standing_objects.py <서베이 폴더> [...] [--extrinsics <hand-eye yaml>]
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_plus_plus_core.head_scan import head_fk, standing_pose, survey  # noqa: E402

MESH_DIR = Path.home() / "rl_ws/sim2real/assets/meshes"
#: 물체 → (메시, 원점 높이 m, OpenCV hue) — sim2real config/objects.d 의 값(2026-10-13 확인)
OBJECTS = {"source240b_blue": ("source240b.obj", 0.085, 100.0), "shaker_c_orange": ("shaker_c.obj", 0.065, 12.0)}


def main() -> int:
    args = sys.argv[1:]
    extrinsics = head_fk.EXTRINSICS_YAML
    if "--extrinsics" in args:
        i = args.index("--extrinsics")
        extrinsics = Path(args[i + 1])
        del args[i:i + 2]
    entries = []
    for d in map(Path, args):
        entries += [(d, r) for r in json.load(open(d / "survey.json"))["records"]]
    fresh = survey.drop_stale([r for _, r in entries])
    for name, (mesh_file, h0, hue) in OBJECTS.items():
        mesh = trimesh.load(MESH_DIR / mesh_file, force="mesh")
        observations = []
        for (d, rec), rec_fresh in zip(entries, fresh):
            raw = rec_fresh["raw_cam"].get(name)
            if raw is None or rec["frame"] is None:
                continue
            z = np.load(d / rec["frame"])
            pan, tilt = rec["measured"]
            xy = (head_fk.t_base_cam(pan, tilt, extrinsics) @ np.r_[raw["pos"], 1])[:2]
            observations.append(standing_pose.StandingObs(z["rgb"], z["K"], pan, tilt, (float(xy[0]), float(xy[1]))))
        est = standing_pose.estimate_standing(observations, mesh.vertices, mesh.faces, h0, hue, extrinsics)
        print(f"{name}: " + ("관측 부족 — 값 없음" if est is None else
              f"x {est.x:.4f} y {est.y:.4f} z {est.z:.4f} · 사용 {est.n_used} 제외 {est.n_rejected} · 흩어짐 {1000 * est.spread_m:.1f} mm · 평균 IoU {est.mean_iou:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
