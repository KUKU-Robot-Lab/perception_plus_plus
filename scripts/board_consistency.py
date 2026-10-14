#!/usr/bin/env python3
"""capture_poses.py 가 저장한 폴더에서 ChArUco 보드의 자세 간 일관성(= 목 FK·hand-eye 오차)을 hand-eye 후보별로 비교한다. 목은 안 움직인다.

    python3 scripts/board_consistency.py <폴더> [--extrinsics a.yaml b.yaml ...] [--min-corners 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_plus_plus_core.head_scan import board_consistency as bc, head_fk  # noqa: E402
from perception_plus_plus_core.head_scan.board_data import load_detections, pose_obs  # noqa: E402

CANDIDATES = {
    "현행(v2)": head_fk.EXTRINSICS_YAML,
    "v3": head_fk.CONFIG_DIR / "head_extrinsics_bolt_v3_candidate.yaml",
    "v4": head_fk.CONFIG_DIR / "head_extrinsics_joint_v4_candidate.yaml",
    "v5": head_fk.CONFIG_DIR / "head_extrinsics_joint_v5_candidate.yaml",
    "v6(CAD 볼트만)": head_fk.CONFIG_DIR / "head_extrinsics_cad_v6_candidate.yaml",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--extrinsics", nargs="*", type=Path, default=None)
    ap.add_argument("--min-corners", type=int, default=6)
    args = ap.parse_args()
    cands = {p.stem: p for p in args.extrinsics} if args.extrinsics else {k: v for k, v in CANDIDATES.items() if v.exists()}
    dets = load_detections(args.folder, args.min_corners, next(iter(cands.values())))
    if len({d["rec"]["index"] for d in dets}) < 2:
        print("❌ 코너가 충분한 자세가 2개 미만", file=sys.stderr)
        return 1
    for name, yaml_path in cands.items():
        print(f"== {name}")
        for scope, subset in (("전체", dets), ("보드 A", [d for d in dets if d["board"] == "A"]), ("보드 B", [d for d in dets if d["board"] == "B"])):
            if len({d["rec"]["index"] for d in subset}) < 2:
                print(f"   {scope}: 자세 2개 미만 — 건너뜀")
                continue
            report = bc.consistency([pose_obs(d, yaml_path) for d in subset])
            at = {z: report.rms_by_z[round(z, 4)] for z in (0.200, 0.205)}
            print(f"   {scope}: 공통 코너 {report.n_common} · 자세 간 RMS 최소 {1000 * report.rms_at_best_m:.2f} mm (보드 높이 {report.best_z:.4f}) · "
                  f"z 0.200 → {1000 * at[0.200]:.2f} mm · z 0.205 → {1000 * at[0.205]:.2f} mm")
        for d in dets:
            scale, resid = bc.grid_scale(pose_obs(d, yaml_path, offset_b=False), 0.200)
            z_unit = bc.z_for_unit_scale(pose_obs(d, yaml_path, offset_b=False))
            print(f"   자세 {d['rec']['commanded']} 보드 {d['board']}: 격자 배율(z 0.200) {scale:.4f} · 잔차 {1000 * resid:.2f} mm · 배율 1.000 이 되는 높이 {z_unit:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
