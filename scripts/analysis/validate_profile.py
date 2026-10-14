"""보정 프로필을 프로필 만들 때 쓰지 않은 서베이로 검증한다(독립 검증). 목은 안 움직인다.

사용: python3 scripts/analysis/validate_profile.py --profile v9_1014_anchored_z --survey <서베이 폴더> [--hand-eye-yaml <plain 프로필 hand-eye 확인용>]
세 가지를 본다:
  ① 볼트: 프로필 hand-eye 만 / + 각도별 보정, 그 각도의 볼트 평균 오차(mm). 표에 없는 각도는 건너뛴다.
  ② 보드 A·B 코너: 각도별 보정을 넣고 자세 간 일관성 RMS(mm, 상판 0.205).
  ③ FP++: sim2real 노드와 같은 변환(앵커·known z 포함)으로 base 좌표를 내고 낡은 값은 빼서 자세 간 흩어짐.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent), "/home/user/rl_ws/sim2real/scripts",
                "/home/user/rl_ws/sim2real/scripts/nodes"]
import bolt_prior_detect as B  # noqa: E402
from object_pose_node import PoseConverter  # noqa: E402
from object_registry import DEFAULT_REGISTRY, load_registry  # noqa: E402

from perception_plus_plus_core.head_scan import board_consistency as bc, calib_profile as cpm, head_fk, refit, survey  # noqa: E402

Z = B.T.TABLE_TOP_Z


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--survey", type=Path, required=True)
    args = ap.parse_args()
    profile = cpm.load_profile(args.profile)
    hand_eye = Path("/tmp") / f"validate_{profile.name}.yaml"
    cpm.write_extrinsics_yaml(profile, hand_eye)
    recs = json.load(open(args.survey / "survey.json"))["records"]
    holes = B.T.cad_holes()

    print(f"== ① 볼트 (프로필 {profile.name}, 서베이 {args.survey.name})")
    corrected = {}
    for r in recs:
        pan, tilt = r["measured"]
        z = np.load(args.survey / r["frame"])
        img, obj = B.detect(cv2.cvtColor(z["rgb"], cv2.COLOR_RGB2GRAY), z["K"], head_fk.t_base_cam(pan, tilt, hand_eye), holes)
        entry = cpm.lookup(profile, pan, tilt)
        if len(img) < 4 or entry is None:
            print(f"  pan {pan:+6.1f}: 볼트 {len(img)}개, {'프로필 표에 없는 각도' if entry is None else '볼트 부족'} — 건너뜀")
            continue
        det = refit.Detection(pan, tilt, "bolt", np.arange(len(img)), img, z["K"])
        est = refit._ground_xy(det, refit.Params(), Z, hand_eye)
        before = np.linalg.norm(est - obj[:, :2], axis=1).mean() * 1000
        after = np.linalg.norm(est + np.array(entry.bias_xy_m) - obj[:, :2], axis=1).mean() * 1000
        corrected[r["index"]] = entry.bias_xy_m
        print(f"  pan {pan:+6.1f}: 볼트 {len(img):2d}개 · hand-eye 만 {before:.2f} mm → 각도별 보정 후 {after:.2f} mm (표 값 {1000 * entry.bias_xy_m[0]:+.2f},{1000 * entry.bias_xy_m[1]:+.2f})")

    print("== ② 보드 코너 자세 간 일관성(각도별 보정 포함)")
    per_pose = []
    for r in recs:
        pan, tilt = r["measured"]
        z = np.load(args.survey / r["frame"])
        entry = cpm.lookup(profile, pan, tilt)
        if entry is None:
            continue
        bias = np.array(entry.bias_xy_m)
        pose = {}
        for ids, uv in bc.detect_boards(z["rgb"], 6):
            t = head_fk.t_base_cam(pan, tilt, hand_eye)
            label = "B" if bc._plane_xy(bc.PoseObs(t, z["K"], np.array([0]), uv.mean(axis=0, keepdims=True)), 0.2)[0, 1] < -0.2 else "A"
            xy = refit._ground_xy(refit.Detection(pan, tilt, label, ids, uv, z["K"]), refit.Params(), Z, hand_eye) + bias
            pose.update({(label, int(i)): v for i, v in zip(ids, xy)})
        per_pose.append(pose)
    keys = {}
    for p in per_pose:
        for k, v in p.items():
            keys.setdefault(k, []).append(v)
    sq = [np.sum((np.array(v) - np.mean(v, axis=0)) ** 2, axis=1) for v in keys.values() if len(v) >= 2]
    print(f"  공통 코너 {len(sq)}개, 자세 {len(per_pose)}곳: RMS {np.sqrt(np.mean(np.concatenate(sq))) * 1000:.2f} mm" if sq else "  비교할 공통 코너 없음")

    print("== ③ FP++ (sim2real 노드와 같은 변환)")
    reg = load_registry(DEFAULT_REGISTRY)
    names = ("source240b_blue", "shaker_c_orange")
    conv = PoseConverter(reg, list(names), calib_profile=args.profile)
    for n in names:
        pts = []
        for r, rf in zip(recs, survey.drop_stale(recs)):
            raw = rf["raw_cam"].get(n)
            if not raw:
                if n in r["raw_cam"]:
                    print(f"  {n[:6]} pan {r['measured'][0]:+6.1f}: 낡은 값(이전 자세와 위치가 같다) — 제외")
                continue
            try:
                p = conv.convert(n, np.array(raw["pos"]), np.array(raw["quat_wxyz"]), tuple(r["measured"]))[0]
            except cpm.UncalibratedAngle:
                print(f"  {n[:6]} pan {r['measured'][0]:+6.1f}: 프로필 표에 없는 각도 — 발행 보류")
                continue
            pts.append(p)
            print(f"  {n[:6]} pan {r['measured'][0]:+6.1f}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
        if len(pts) > 1:
            a = np.array(pts)
            print(f"  {n[:6]} 자세 {len(a)}곳 흩어짐 표준편차 ({1000 * a[:, 0].std(ddof=1):.1f}, {1000 * a[:, 1].std(ddof=1):.1f}, {1000 * a[:, 2].std(ddof=1):.1f}) mm · 최대 이탈 {1000 * np.linalg.norm(a[:, :2] - a[:, :2].mean(0), axis=1).max():.1f} mm")


if __name__ == "__main__":
    main()
