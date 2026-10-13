"""볼트(CAD 구멍) 관측으로 목 FK 의 상수 hand-eye 보정을 맞추고, 프레임 하나씩 빼며 검증한다(교차 검증).

사용: python3 scripts/analysis/handeye_calibrate.py <서베이 폴더> [<서베이 폴더> ...]
모델 A: 카메라 쪽 상수 6자유도(회전 3 + 이동 3). 모델 B: A + pan·tilt 인코더 영점 오프셋 2개.
검증: (1) 뺀 프레임의 볼트 오차 (2) 보정 변환으로 옮긴 컵 위치의 자세 간 흩어짐(볼트와 독립인 관측).
sim2real calib 모듈은 경로로 불러 읽기만 한다. 결과 값은 제안용이며 head_extrinsics.yaml 은 고치지 않는다.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path[:0] = ["/home/user/rl_ws/sim2real/scripts", "/home/user/rl_ws/sim2real/scripts/calib",
                "/home/user/rl_ws/perception_plus_plus"]
import table_cad_extrinsics as T  # noqa: E402

from perception_plus_plus_core.head_scan import head_fk, survey  # noqa: E402

MIN_BOLTS = 8


def backproject(t_bc, k, uv, z_plane):
    d = np.c_[(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(uv))]
    dr = d @ t_bc[:3, :3].T
    o = t_bc[:3, 3]
    return o + dr * ((z_plane - o[2]) / dr[:, 2])[:, None]


def load_frames(dirs):
    holes = T.cad_holes()
    frames, cups = [], []
    entries = []  # 폴더 순서대로 이어 붙인다 — 낡은 값 판정이 폴더 경계를 넘어 일어나야 한다
    for d in map(Path, dirs):
        entries += [(d, rec) for rec in json.load(open(d / "survey.json"))["records"]]
    fresh = survey.drop_stale([rec for _, rec in entries])
    for (d, rec), rec_fresh in zip(entries, fresh):
        if rec["frame"] is None:
            continue
        z = np.load(d / rec["frame"])
        gray = cv2.cvtColor(z["rgb"], cv2.COLOR_RGB2GRAY)
        pan, tilt = rec["measured"]
        for name, raw in rec_fresh["raw_cam"].items():
            cups.append((name, pan, tilt, np.array(raw["pos"])))
        try:
            _, used, img, _, _ = T.solve(holes, gray, z["K"], head_fk.t_base_cam(pan, tilt))
        except RuntimeError:
            continue
        if len(used) >= MIN_BOLTS:
            frames.append({"tag": f"{d.name}[{rec['index']}]", "pan": pan, "tilt": tilt, "k": z["K"], "img": img,
                           "obj": np.array([holes[i].xyz for i in used])})
    return frames, cups


def t_base_cam(x, pan, tilt, model):
    off_pan, off_tilt = (x[6], x[7]) if model == "B" else (0.0, 0.0)
    corr = np.eye(4)
    corr[:3, :3] = cv2.Rodrigues(x[:3])[0]
    corr[:3, 3] = x[3:6]
    return head_fk.t_base_cam(pan + np.degrees(off_pan), tilt + np.degrees(off_tilt)) @ corr


def bolt_err_mm(x, f, model):
    e = backproject(t_base_cam(x, f["pan"], f["tilt"], model), f["k"], f["img"], T.TABLE_TOP_Z)[:, :2] - f["obj"][:, :2]
    return e * 1000


def cup_resid_mm(x, cups, model, weight):
    """같은 컵은 늘 같은 자리 — 자세별 base 위치가 그 컵의 평균에서 벗어난 만큼(mm)을 잔차로 둔다."""
    out = []
    for name in sorted({c[0] for c in cups}):
        pts = np.array([(t_base_cam(x, p, t, model) @ np.r_[pos, 1])[:3] for n, p, t, pos in cups if n == name])
        if len(pts) > 1:
            out.append(weight * 1000 * (pts - pts.mean(0)).ravel())
    return np.concatenate(out) if out else np.zeros(0)


def fit(frames, model, cups=(), cup_weight=0.0):
    x0 = np.zeros(8 if model == "B" else 6)

    def resid(x):
        parts = [bolt_err_mm(x, f, model).ravel() for f in frames]
        if cup_weight > 0:
            parts.append(cup_resid_mm(x, list(cups), model, cup_weight))
        return np.concatenate(parts)
    return least_squares(resid, x0).x


def cup_spread_mm(x, cups, model):
    out = {}
    for name in sorted({c[0] for c in cups}):
        pts = np.array([(t_base_cam(x, p, t, model) @ np.r_[pos, 1])[:3] for n, p, t, pos in cups if n == name])
        out[name] = (len(pts), 1000 * np.linalg.norm(pts - pts.mean(0), axis=1).max(), pts.mean(0))
    return out


def mean_err(x, f, model):
    return np.linalg.norm(bolt_err_mm(x, f, model), axis=1).mean()


def write_candidate(path, x, note):
    """보정된 `T_neck_cam' = T_neck_cam ∘ corr` 를 head_extrinsics.yaml 과 같은 형식의 후보 파일로 쓴다(원본은 고치지 않는다)."""
    corr = np.eye(4)
    corr[:3, :3] = cv2.Rodrigues(x[:3])[0]
    corr[:3, 3] = x[3:6]
    matrix = head_fk.load_neck_to_camera() @ corr
    rows = "\n".join("    - [" + ", ".join(f"{v:+.9f}" for v in row) + "]" for row in matrix)
    Path(path).write_text(f"# hand-eye 후보 — {note}\n# T_neck_cam' = T_neck_cam(v2) ∘ corr. 적용은 사용자 소유 sim2real/config 에서 합의 후.\n"
                          f"neck_to_camera:\n  matrix:\n{rows}\n", encoding="utf-8")


def main():
    args = sys.argv[1:]
    out = None
    if "--write" in args:
        i = args.index("--write")
        out = args[i + 1]
        del args[i:i + 2]
    frames, cups = load_frames(args)
    print(f"볼트 프레임 {len(frames)}개, 컵 관측 {len(cups)}개 (낡은 값 제외)")
    model = "A"  # 모델 B(영점 오프셋 추가)는 A 와 같은 결과로 수렴해 식별되지 않았다(2026-10-13)
    x0 = np.zeros(6)
    base = [mean_err(x0, f, model) for f in frames]
    for w in (0.0, 0.5, 1.0):
        loo = [mean_err(fit(frames[:i] + frames[i + 1:], model, cups, w), f, model) for i, f in enumerate(frames)]
        x = fit(frames, model, cups, w)
        full = [mean_err(x, f, model) for f in frames]
        print(f"\n컵 가중치 {w}: 볼트 평균 {np.mean(base):.1f} → 전체 맞춤 {np.mean(full):.1f} (최대 {max(full):.1f}) "
              f"· 교차검증 {np.mean(loo):.1f} (최대 {max(loo):.1f}) mm")
        print("  회전 deg", np.round(np.degrees(x[:3]), 3).tolist(), "이동 mm", np.round(x[3:6] * 1000, 2).tolist())
        for name, (n, spread, mean) in cup_spread_mm(x, cups, model).items():
            print(f"  컵 {name}: 관측 {n}개 · 평균에서 최대 {spread:.1f} mm · 평균 {np.round(mean, 4).tolist()}")
    if out:
        write_candidate(out, x, f"볼트 {len(frames)}프레임 + 컵 일관성(가중치 1.0) 맞춤, 교차검증 {np.mean(loo):.1f} mm")
        print(f"후보 파일 기록: {out}")
    for f, b in zip(frames, base):
        print(f"  {f['tag']:>22} pan {f['pan']:+7.2f} tilt {f['tilt']:+6.2f}: 보정 전 {b:5.1f} mm")


if __name__ == "__main__":
    main()
