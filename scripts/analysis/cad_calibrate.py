"""CAD 볼트만으로 hand-eye 보정을 맞춘다(보드 없이). 짝짓기는 사전 자세로 못 박는다(`bolt_prior_detect`).

사용: python3 scripts/analysis/cad_calibrate.py --surveys <서베이 폴더>... --captures <capture_poses 폴더>... [--prior <yaml>] [--write <후보.yaml>]
미지수: hand-eye 보정 6(회전 3, 이동 3, `T_neck_cam(v2) ∘ corr`). 잔차: 볼트를 상판 평면에 얹은 위치와 CAD 구멍 위치의 차(mm).
검증: 프레임 하나씩 빼고 맞춘 뒤 그 프레임의 볼트 오차. 보드는 쓰지 않으니 보드 일관성(board_consistency.py)이 독립 검증이 된다.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path[:0] = ["/home/user/rl_ws/perception_plus_plus", "/home/user/rl_ws/perception_plus_plus/scripts/analysis"]
import bolt_prior_detect as B  # noqa: E402
import handeye_calibrate as H  # noqa: E402

from perception_plus_plus_core.head_scan import head_fk, refit  # noqa: E402

T = B.T
MIN_BOLTS = 5


def load_frames(surveys, captures, prior):
    holes = T.cad_holes()
    sources = []
    for d in map(Path, surveys):
        sources += [(d, f"{d.name}[{r['index']}]", r["frame"], r["measured"]) for r in json.load(open(d / "survey.json"))["records"] if r["frame"]]
    for d in map(Path, captures):
        sources += [(d, f"{d.name}[{r['index']}]", f"pose_{r['index']}_0.npz", r["measured"]) for r in json.load(open(d / "poses.json"))["records"]]
    frames = []
    for d, tag, name, (pan, tilt) in sources:
        z = np.load(d / name)
        img, obj = B.detect(cv2.cvtColor(z["rgb"], cv2.COLOR_RGB2GRAY), z["K"], head_fk.t_base_cam(pan, tilt, prior), holes)
        if len(img) >= MIN_BOLTS:
            frames.append({"tag": tag, "pan": pan, "tilt": tilt, "k": z["K"], "img": img, "obj": obj})
    return frames


def residual(x, frames):
    params = refit.Params(rot=x[:3], trans=x[3:6])
    out = []
    for f in frames:
        det = refit.Detection(f["pan"], f["tilt"], "bolt", np.arange(len(f["img"])), f["img"], f["k"])
        out.append(((refit._ground_xy(det, params, T.TABLE_TOP_Z, head_fk.EXTRINSICS_YAML) - f["obj"][:, :2]) * 1000).ravel())
    return np.concatenate(out)


def fit(frames, x0=None):
    return least_squares(residual, np.zeros(6) if x0 is None else x0, args=(frames,)).x


def mean_err(x, frames):
    return float(np.linalg.norm(residual(x, frames).reshape(-1, 2), axis=1).mean())


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--surveys", nargs="*", default=[])
    ap.add_argument("--captures", nargs="*", default=[])
    ap.add_argument("--prior", type=Path, default=head_fk.CONFIG_DIR / "head_extrinsics_joint_v5_candidate.yaml")
    ap.add_argument("--write", default=None)
    args = ap.parse_args()
    frames = load_frames(args.surveys, args.captures, args.prior)
    print(f"볼트 프레임 {len(frames)}개(볼트 {min(len(f['img']) for f in frames)}~{max(len(f['img']) for f in frames)}개씩) · tilt {sorted({round(f['tilt']) for f in frames})}")
    x = fit(frames)
    print(f"전체 맞춤: 볼트 평균 오차 {mean_err(x, frames):.2f} mm · 회전 mrad {np.round(x[:3] * 1000, 2).tolist()} · 이동 mm {np.round(x[3:] * 1000, 2).tolist()}")
    held = []
    for i, f in enumerate(frames):
        xi = fit(frames[:i] + frames[i + 1:], x)
        held.append(mean_err(xi, [f]))
    by_tilt = {t: np.mean([e for e, f in zip(held, frames) if round(f["tilt"]) == t]) for t in sorted({round(f["tilt"]) for f in frames})}
    print(f"교차검증(프레임 하나 빼고): 볼트 평균 {np.mean(held):.2f} mm (최대 {max(held):.2f}) · tilt 별 " + ", ".join(f"{t}: {v:.2f}" for t, v in by_tilt.items()))
    if args.write:
        H.write_candidate(args.write, x, f"CAD 볼트만(보드 없이, 사전 자세 짝짓기) {len(frames)}프레임, 교차검증 볼트 {np.mean(held):.2f} mm")
        print("후보 파일 기록:", args.write)


if __name__ == "__main__":
    main()
