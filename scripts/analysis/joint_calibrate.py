"""볼트 재투영 + 컵 실루엣 일치를 함께 최적화해 hand-eye 상수 보정(6자유도)과 컵 x, y 를 구하고, 프레임 하나씩 빼며 검증한다.

사용: python3 scripts/analysis/joint_calibrate.py <서베이 폴더> [...] [--write <후보.yaml>]
미지수: 카메라 쪽 상수 보정 6 + 컵마다 (x, y). z 는 상판+원점 높이로 고정, 컵은 똑바로 선 것으로 본다.
비용 = 볼트 오차 제곱 평균(mm²) + 컵 (1-IoU)*100 제곱 평균. 검증: 한 자세를 빼고 맞춘 뒤 그 자세의 볼트 오차와 컵 위치 편차.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize

sys.path[:0] = ["/home/user/rl_ws/perception_plus_plus", "/home/user/rl_ws/perception_plus_plus/scripts/analysis"]
import handeye_calibrate as H  # noqa: E402

from perception_plus_plus_core.head_scan import head_fk, silhouette_fit as S, survey  # noqa: E402

T = H.T
MESH_DIR = Path("/home/user/rl_ws/sim2real/assets/meshes")
OBJECTS = {"source240b_blue": ("source240b.obj", 0.085, 100.0), "shaker_c_orange": ("shaker_c.obj", 0.065, 12.0)}
SIL_WEIGHT = 1.0
MIN_IOU_FOR_FIT = 0.85  # 마스크가 불량한 프레임(tilt -30 등)은 컵 항에서 뺀다


def load(dirs):
    holes = T.cad_holes()
    entries = []
    for d in map(Path, dirs):
        entries += [(d, r) for r in json.load(open(d / "survey.json"))["records"]]
    fresh = survey.drop_stale([r for _, r in entries])
    meshes = {n: trimesh.load(MESH_DIR / f, force="mesh") for n, (f, _, _) in OBJECTS.items()}
    frames = []
    for (d, r), rf in zip(entries, fresh):
        if r["frame"] is None:
            continue
        z = np.load(d / r["frame"])
        rgb, k = z["rgb"], z["K"]
        pan, tilt = r["measured"]
        try:
            _, used, img, _, _ = T.solve(holes, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), k, head_fk.t_base_cam(pan, tilt))
        except RuntimeError:
            used = []
        if len(used) < H.MIN_BOLTS:
            continue
        t0 = head_fk.t_base_cam(pan, tilt, H.Path("/home/user/rl_ws/perception_plus_plus/config/robot_camera_extrinsics/head_extrinsics_bolt_v3_candidate.yaml"))
        cups = {}
        for name, raw in rf["raw_cam"].items():
            h0, hue = OBJECTS[name][1:]
            xy = (t0 @ np.r_[raw["pos"], 1])[:2]
            uv, _ = S.project_points(np.array([[xy[0], xy[1], T.TABLE_TOP_Z + h0]]), t0, k)
            mask = S.largest_blob_near(S.hue_mask(rgb, hue), tuple(uv[0]))
            if mask.sum() >= 200:
                cups[name] = {"mask": mask, "xy0": xy}
        frames.append({"tag": f"{d.name}[{r['index']}]", "pan": pan, "tilt": tilt, "k": k, "img": img,
                       "obj": np.array([holes[i].xyz for i in used]), "cups": cups})
    return frames, meshes


def cost_terms(x, frames, meshes, names, cup_xy):
    bolt, sil = [], []
    for f in frames:
        tbc = H.t_base_cam(x, f["pan"], f["tilt"], "A")
        bolt.append((H.bolt_err_mm(x, f, "A") ** 2).sum(1).mean())
        for name, c in f["cups"].items():
            m = meshes[name]
            px, py = cup_xy[names.index(name)]
            sil_img = S.render_silhouette(m.vertices + np.array([px, py, T.TABLE_TOP_Z + OBJECTS[name][1]]), m.faces, tbc, f["k"], c["mask"].shape)
            sil.append(((1 - S.iou(sil_img, c["mask"])) * 100) ** 2)
    return float(np.mean(bolt)), float(np.mean(sil)) if sil else 0.0


def fit(frames, meshes, names, x_init, xy_init):
    def cost(p):
        b, s = cost_terms(p[:6], frames, meshes, names, p[6:].reshape(-1, 2))
        return b + SIL_WEIGHT * s
    scale = np.r_[np.full(3, 0.01), np.full(3, 0.005), np.full(len(names) * 2, 0.002)]
    p0 = np.r_[x_init, np.ravel(xy_init)]
    res = minimize(cost, p0, method="Powell", options={"xtol": 1e-4, "ftol": 1e-5, "maxfev": 1500,
                                                       "direc": np.diag(scale)})
    return res.x[:6], res.x[6:].reshape(-1, 2), res.fun


def main():
    args = sys.argv[1:]
    out = None
    if "--write" in args:
        i = args.index("--write")
        out = args[i + 1]
        del args[i:i + 2]
    frames, meshes = load(args)
    names = sorted(OBJECTS)
    xy_init = np.array([np.mean([f["cups"][n]["xy0"] for f in frames if n in f["cups"]], axis=0) for n in names])
    x0 = np.zeros(6)
    print(f"프레임 {len(frames)}개 · 컵 마스크 {sum(len(f['cups']) for f in frames)}개")
    full_x, full_xy, _ = fit(frames, meshes, names, x0, xy_init)
    print("전체 맞춤 회전 deg", np.round(np.degrees(full_x[:3]), 3).tolist(), "이동 mm", np.round(full_x[3:] * 1000, 2).tolist())
    for n, p in zip(names, full_xy):
        print(f"  컵 {n} 위치 {np.round(p, 4).tolist()}")
    loo_bolt, loo_cup = [], []
    for i, f in enumerate(frames):
        rest = frames[:i] + frames[i + 1:]
        xi, xyi, _ = fit(rest, meshes, names, full_x, full_xy)
        eb = H.mean_err(xi, f, "A")
        line = f"  {f['tag']:>22} pan {f['pan']:+7.2f}: 볼트 {eb:4.1f} mm"
        loo_bolt.append(eb)
        for n, c in f["cups"].items():
            pos = np.array([f["cups"][n]["xy0"]])  # FPP→FK v3 와 비교용
            m = meshes[n]
            res = S.fit_xy(m.vertices, m.faces, T.TABLE_TOP_Z + OBJECTS[n][1], H.t_base_cam(xi, f["pan"], f["tilt"], "A"), f["k"],
                           c["mask"], tuple(xyi[names.index(n)]), search_m=0.02)
            dev = 1000 * np.hypot(res.x - xyi[names.index(n), 0], res.y - xyi[names.index(n), 1])
            loo_cup.append(dev)
            line += f" · {n[:6]} 편차 {dev:4.1f} mm (IoU {res.iou:.2f})"
        print(line)
    print(f"교차검증: 볼트 평균 {np.mean(loo_bolt):.1f} mm (최대 {max(loo_bolt):.1f}) · 컵 편차 평균 {np.mean(loo_cup):.1f} mm (최대 {max(loo_cup):.1f})")
    if out:
        H.write_candidate(out, full_x, f"볼트 + 컵 실루엣 결합 맞춤 {len(frames)}프레임, 교차검증 볼트 {np.mean(loo_bolt):.1f} mm")
        print("후보 파일 기록:", out)


if __name__ == "__main__":
    main()
