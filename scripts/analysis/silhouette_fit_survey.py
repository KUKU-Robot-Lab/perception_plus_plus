"""서베이 프레임마다 컵 x, y 를 CAD 실루엣 맞춤으로 다시 구하고, FPP 값과 자세 간 흩어짐을 비교한다.
사용: python3 scripts/analysis/silhouette_fit_survey.py <서베이 폴더> [...]   (카메라 자세는 hand-eye v3 후보 + 촬영 각도 FK)
z 는 상판 0.205 + 원점 높이로 고정하고 자세는 똑바로 선 것으로 본다. 결과 영상은 <폴더>/fit_<번호>_<물체>.png."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path[:0] = ["/home/user/rl_ws/perception_plus_plus"]
from perception_plus_plus_core.head_scan import head_fk, silhouette_fit as S, survey  # noqa: E402

CAND = Path("/home/user/rl_ws/perception_plus_plus/config/robot_camera_extrinsics/head_extrinsics_bolt_v3_candidate.yaml")
MESH_DIR = Path("/home/user/rl_ws/sim2real/assets/meshes")
TABLE_Z = 0.205
#: 물체 → (메시, CAD 원점 높이 m, OpenCV hue 중심) — sim2real config/objects.d 의 값(2026-10-13 확인)
OBJECTS = {"source240b_blue": ("source240b.obj", 0.085, 100.0), "shaker_c_orange": ("shaker_c.obj", 0.065, 12.0)}

meshes = {n: trimesh.load(MESH_DIR / f, force="mesh") for n, (f, _, _) in OBJECTS.items()}
entries = []
for d in map(Path, sys.argv[1:]):
    entries += [(d, r) for r in json.load(open(d / "survey.json"))["records"]]
fresh = survey.drop_stale([r for _, r in entries])
rows = {n: [] for n in OBJECTS}
for (d, r), rf in zip(entries, fresh):
    if r["frame"] is None:
        continue
    z = np.load(d / r["frame"])
    rgb, k = z["rgb"], z["K"]
    pan, tilt = r["measured"]
    t_bc = head_fk.t_base_cam(pan, tilt, CAND)
    for name, raw in rf["raw_cam"].items():
        mesh_file, h0, hue = OBJECTS[name]
        init = (t_bc @ np.r_[raw["pos"], 1])[:2]
        uv, _ = S.project_points(np.array([[init[0], init[1], TABLE_Z + h0]]), t_bc, k)
        mask = S.largest_blob_near(S.hue_mask(rgb, hue), tuple(uv[0]))
        if mask.sum() < 200:
            print(f"{d.name}[{r['index']}] {name}: 마스크 없음 건너뜀")
            continue
        fit = S.fit_xy(meshes[name].vertices, meshes[name].faces, TABLE_Z + h0, t_bc, k, mask, tuple(init))
        rows[name].append((d.name, r["index"], pan, init, fit))
        sil = S.render_silhouette(meshes[name].vertices + np.array([fit.x, fit.y, TABLE_Z + h0]), meshes[name].faces, t_bc, k, mask.shape)
        vis = rgb.copy()[..., ::-1].copy()
        vis[mask] = (0.5 * vis[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        cnts, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), 1)
        cv2.imwrite(str(d / f"fit_{r['index']}_{name}.png"), vis)
for name, lst in rows.items():
    print(name)
    for tag, idx, pan, init, f in lst:
        print(f"  {tag}[{idx}] pan {pan:+7.2f}  FPP→base ({init[0]:.4f},{init[1]:.4f})  실루엣 ({f.x:.4f},{f.y:.4f})  IoU {f.init_iou:.2f}→{f.iou:.2f}")
    for lab, getter in (("FPP", lambda x: x[3]), ("실루엣", lambda x: np.array([x[4].x, x[4].y]))):
        pts = np.array([getter(x) for x in lst])
        dev = 1000 * np.linalg.norm(pts - pts.mean(0), axis=1)
        print(f"  {lab}: 평균 {np.round(pts.mean(0), 4)} · 평균에서 최대 {dev.max():.1f} mm · 표준편차 {np.round(1000 * pts.std(0), 1)} mm")
