#!/usr/bin/env python3
"""보정 프로필 파일을 만든다 — hand-eye 후보 하나 + 목 각도별 CAD 볼트 보정(평행이동). 목은 안 움직인다.

    python3 scripts/make_calib_profile.py --name v6_1014 --hand-eye config/robot_camera_extrinsics/head_extrinsics_cad_v6_candidate.yaml \\
        --surveys <서베이 폴더>... --captures <capture_poses 폴더>... [--out <경로>] [--no-bias] [--lenient]

각도 표는 영상을 찍은 목 자세(명령 pan,tilt)마다 한 항목이다. 같은 명령 자세의 프레임은 폴더를 넘어 합친다.
각도 보정 = 그 영상들에서 CAD 구멍 위치와 (hand-eye 로 구한) 볼트 위치의 평균 차이(상판 높이, mm 단위 평행이동).
--no-bias 면 hand-eye 만 담고 보정은 0 으로 둔다. 쓰는 쪽은 sim2real `scripts/calib_profile.py` 가 읽는다(스키마는 그 파일 머리말).
기본 출력: sim2real/config/calib_profiles/<name>.yaml (새 파일만 만든다. 기존 설정은 건드리지 않는다).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent / "analysis")]
import bolt_prior_detect as B  # noqa: E402

from perception_plus_plus_core.head_scan import calib_profile as cpm, head_fk, refit  # noqa: E402

Z = B.T.TABLE_TOP_Z


def collect(surveys, captures, hand_eye: Path):
    """(명령 자세 → [프레임 기록]) 볼트 검출은 hand-eye 후보를 사전 자세로 쓴다."""
    holes = B.T.cad_holes()
    sources = []
    for d in map(Path, surveys):
        sources += [(d / r["frame"], tuple(r["commanded"]), r["measured"]) for r in json.load(open(d / "survey.json"))["records"] if r["frame"]]
    for d in map(Path, captures):
        sources += [(d / f"pose_{r['index']}_0.npz", tuple(r["commanded"]), r["measured"]) for r in json.load(open(d / "poses.json"))["records"]]
    groups: dict[tuple[float, float], list[dict]] = {}
    for path, commanded, (pan, tilt) in sources:
        z = np.load(path)
        img, obj = B.detect(cv2.cvtColor(z["rgb"], cv2.COLOR_RGB2GRAY), z["K"], head_fk.t_base_cam(pan, tilt, hand_eye), holes)
        if len(img) < 5:
            continue
        det = refit.Detection(pan, tilt, "bolt", np.arange(len(img)), img, z["K"])
        est = refit._ground_xy(det, refit.Params(), Z, hand_eye)
        groups.setdefault(commanded, []).append({"pan": pan, "tilt": tilt, "est": est, "obj": obj[:, :2]})
    return groups


def build(name: str, hand_eye: Path, groups, bias: bool, strict: bool, notes: str) -> cpm.CalibProfile:
    entries = []
    for commanded in sorted(groups):
        frames = groups[commanded]
        diff = np.vstack([f["obj"] - f["est"] for f in frames])
        shift = diff.mean(axis=0) if bias else np.zeros(2)
        resid = float(np.linalg.norm(diff - shift, axis=1).mean() * 1000)
        entries.append(cpm.AngleEntry(float(np.mean([f["pan"] for f in frames])), float(np.mean([f["tilt"] for f in frames])),
                                      (float(shift[0]), float(shift[1])), len(frames), len(diff), resid))
    return cpm.CalibProfile(name=name, neck_to_camera=head_fk.load_neck_to_camera(hand_eye), entries=tuple(entries), strict=strict, notes=notes)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--hand-eye", type=Path, required=True)
    ap.add_argument("--surveys", nargs="*", default=[])
    ap.add_argument("--captures", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--lenient", action="store_true", help="표에 없는 각도도 보정 0 으로 통과(기본은 거부)")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    groups = collect(args.surveys, args.captures, args.hand_eye)
    profile = build(args.name, args.hand_eye, groups, not args.no_bias, not args.lenient,
                    args.notes or f"hand-eye {args.hand_eye.name}, 각도 {len(groups)}개, {'평행이동 보정 포함' if not args.no_bias else '보정 없음'}")
    out = args.out or cpm.PROFILE_DIR / f"{args.name}.yaml"
    if out.exists():
        print(f"❌ {out} 가 이미 있다 — 덮어쓰지 않는다(다른 이름으로)", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    cpm.save_profile(profile, out)
    for e in profile.entries:
        print(f"  pan {e.pan_deg:+7.2f} tilt {e.tilt_deg:+6.2f}: 프레임 {e.n_frames} 볼트 {e.n_bolts:3d} · 보정 ({1000 * e.bias_xy_m[0]:+.2f}, {1000 * e.bias_xy_m[1]:+.2f}) mm · 보정 후 볼트 잔차 {e.bolt_resid_mm:.2f} mm")
    print(f"프로필 기록: {out} (각도 {len(profile.entries)}개, strict={profile.strict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
