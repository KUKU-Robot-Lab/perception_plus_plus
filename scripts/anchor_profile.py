#!/usr/bin/env python3
"""보정 프로필의 출력을 **기존에 검증된 FP++ 좌표계**에 붙인다(프레임 앵커). 목은 안 움직인다.

    python3 scripts/anchor_profile.py --profile v6_1014 --name v6_1014_anchored --surveys <서베이 폴더>... [--ref-pan 0 --ref-tilt -20]

왜: 기존 FP++ 변환(정적 camera 블록, 기준 자세 pan 0 / tilt -20)은 검증이 끝났다. CAD 로 맞춘 프로필은 같은 기준 자세에서도 그것과
mm 단위로 다르다(2026-10-14 실측: x -5~-12, y +7.5, z +3 mm). 정책·궤적이 검증된 좌표계를 쓰므로 기준 자세 출력은 같아야 한다.
방법: 기준 자세 영상(서베이 명령 자세 ref)에서
  ① xy: CAD 볼트 픽셀을 (프로필 hand-eye + 그 각도 보정)과 (기존 정적 camera 블록) 두 방식으로 상판에 얹어, 앞 것을 뒤 것에 맞추는
     강체(z 둘레 회전 + 이동)를 최소제곱으로 구한다.
  ② z: 같은 FP++ 관측을 기존 변환과 프로필 변환으로 base 로 바꿔 z 차이의 평균을 dz 로 둔다.
프로필의 다른 각도 출력도 같은 앵커를 거치므로 모든 각도가 검증된 좌표계로 간다(각도 사이 일관성은 그대로).
출력: 프로필 파일 새 이름(기존 파일을 덮어쓰지 않는다). 검증: 기준 자세 FP++ 출력이 기존과 얼마나 같은지 + 각도 사이 일관성.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent / "analysis"),
                "/home/user/rl_ws/sim2real/scripts", "/home/user/rl_ws/sim2real/scripts/nodes"]
import bolt_prior_detect as B  # noqa: E402
from object_pose_node import PoseConverter  # noqa: E402
from object_registry import DEFAULT_REGISTRY, load_registry  # noqa: E402

from perception_plus_plus_core.head_scan import calib_profile as cpm, head_fk, refit, survey  # noqa: E402

Z = B.T.TABLE_TOP_Z
STATIC_YAML = Path("/home/user/rl_ws/sim2real/config/global_camera_extrinsics.yaml")


def static_camera() -> np.ndarray:
    """기존 정적 camera 블록의 T_base_cam(기준 자세 pan 0 / tilt -20 전용)."""
    cam = yaml.safe_load(STATIC_YAML.read_text(encoding="utf-8"))["camera"]
    w, x, y, z = cam["orientation_wxyz"]
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot, cam["position"]
    return t


def plane_xy(t_base_cam: np.ndarray, k: np.ndarray, uv: np.ndarray, z: float) -> np.ndarray:
    ray = np.column_stack([(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(uv))]) @ t_base_cam[:3, :3].T
    return (t_base_cam[:3, 3] + ((z - t_base_cam[2, 3]) / ray[:, 2])[:, None] * ray)[:, :2]


def rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray]:
    """src → dst 강체(z 둘레 회전 rad, 이동)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    rot = u @ np.diag([1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.arctan2(rot[1, 0], rot[0, 0])), mu_d - rot @ mu_s


def ref_records(surveys, ref_pan: float, ref_tilt: float, tol: float = 1.0):
    for d in map(Path, surveys):
        recs = json.load(open(d / "survey.json"))["records"]
        for r, rf in zip(recs, survey.drop_stale(recs)):
            if r["frame"] and abs(r["measured"][0] - ref_pan) <= tol and abs(r["measured"][1] - ref_tilt) <= tol:
                yield d, r, rf


def compute_anchor(profile, profile_path, surveys, ref_pan: float = 0.0, ref_tilt: float = -20.0, log=print):
    """기준 자세 영상에서 프로필 출력을 기존 검증 좌표계에 붙이는 앵커를 구한다 → (FrameAnchor, 통계 dict).

    profile_path 는 디스크에 저장된 (앵커 없는) 프로필 경로(z 비교에 노드와 같은 변환을 쓰려고). surveys 에 기준 자세 프레임이 있어야 한다.
    FP++ 관측이 하나도 없으면 dz 는 0 으로 두고 경고한다.
    """
    hand_eye = Path("/tmp") / f"anchor_hand_eye_{profile.name}.yaml"
    cpm.write_extrinsics_yaml(profile, hand_eye)
    t_static, holes = static_camera(), B.T.cad_holes()
    src, dst, n_frames = [], [], 0
    for d, r, _ in ref_records(surveys, ref_pan, ref_tilt):
        z = np.load(d / r["frame"])
        pan, tilt = r["measured"]
        img, _ = B.detect(cv2.cvtColor(z["rgb"], cv2.COLOR_RGB2GRAY), z["K"], head_fk.t_base_cam(pan, tilt, hand_eye), holes)
        if len(img) < 5:
            continue
        det = refit.Detection(pan, tilt, "bolt", np.arange(len(img)), img, z["K"])
        src.append(refit._ground_xy(det, refit.Params(), Z, hand_eye) + cpm.bias_for(profile, pan, tilt)[:2])
        dst.append(plane_xy(t_static, z["K"], img, Z))
        n_frames += 1
    if not src:
        raise RuntimeError(f"기준 자세(pan {ref_pan}, tilt {ref_tilt}) 영상에서 볼트 5개 이상이 잡힌 프레임이 없다 — 앵커를 못 구한다")
    src, dst = np.vstack(src), np.vstack(dst)
    rot, trans = rigid_fit(src, dst)
    fit_resid = np.linalg.norm(src @ np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]]) + trans - dst, axis=1)
    log(f"기준 자세 영상 {n_frames}장, 볼트 점 {len(src)}개: 프로필 → 기존 정적 좌표계 강체 맞춤 회전 {np.degrees(rot):+.3f}° 이동 ({1000 * trans[0]:+.2f}, {1000 * trans[1]:+.2f}) mm "
        f"· 맞춤 잔차 평균 {1000 * fit_resid.mean():.2f} mm (맞추기 전 평균 차이 {1000 * np.linalg.norm(src - dst, axis=1).mean():.2f} mm)")

    # z: 같은 FP++ 관측의 기존 변환 대 프로필 변환
    reg = load_registry(DEFAULT_REGISTRY)
    names = ("source240b_blue", "shaker_c_orange")
    old, new = PoseConverter(reg, list(names)), PoseConverter(reg, list(names), calib_profile=profile_path)
    zs = []
    for _, r, rf in ref_records(surveys, ref_pan, ref_tilt):
        for n in names:
            raw = rf["raw_cam"].get(n)
            if raw:
                pos, quat = np.array(raw["pos"]), np.array(raw["quat_wxyz"])
                zs.append(old.convert(n, pos, quat)[0][2] - new.convert(n, pos, quat, tuple(r["measured"]))[0][2])
    if zs:
        dz = float(np.mean(zs))
        log(f"z: FP++ 관측 {len(zs)}개, 기존 − 프로필 평균 {1000 * dz:+.2f} mm (표준편차 {1000 * np.std(zs):.2f})")
    else:
        dz = 0.0
        log("⚠ 기준 자세에서 FP++ 관측이 없어 dz 를 0 으로 둔다(known z 를 쓰면 영향이 작다)")
    anchor = cpm.FrameAnchor(float(np.degrees(rot)), (float(trans[0]), float(trans[1])), dz,
                             f"기존 정적 camera 블록 기준 자세(pan {ref_pan}/tilt {ref_tilt}), 볼트 {len(src)}점")
    return anchor, {"n_frames": n_frames, "n_bolts": len(src), "fit_resid_mm": float(1000 * fit_resid.mean()), "n_z_obs": len(zs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--surveys", nargs="+", required=True)
    ap.add_argument("--known-z", type=float, default=None, metavar="TABLE_TOP_M",
                    help="상판 높이(m, 줄자 확정 0.205)를 주면 정지 pose 의 z 를 상판 + 원점 높이로 맞춘다(허용 ±15 mm 안일 때만)")
    ap.add_argument("--ref-pan", type=float, default=0.0)
    ap.add_argument("--ref-tilt", type=float, default=-20.0)
    args = ap.parse_args()
    profile = cpm.load_profile(args.profile)
    anchor, _ = compute_anchor(profile, args.profile, args.surveys, args.ref_pan, args.ref_tilt)
    anchored = dataclasses.replace(profile, name=args.name, anchor=anchor, notes=profile.notes + " + 기존 FP++ 좌표계 앵커" + (" + 아는 z" if args.known_z else ""),
                                   known_z=cpm.KnownZ(args.known_z, 0.015) if args.known_z else None)
    out = cpm.PROFILE_DIR / f"{args.name}.yaml"
    if out.exists():
        print(f"❌ {out} 가 이미 있다 — 덮어쓰지 않는다", file=sys.stderr)
        return 2
    cpm.save_profile(anchored, out)
    print("프로필 기록:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
