#!/usr/bin/env python3
"""capture_poses.py 가 저장한 폴더에서 ChArUco 보드의 자세 간 일관성(= 목 FK·hand-eye 오차)을 hand-eye 후보별로 비교한다. 목은 안 움직인다.

    python3 scripts/board_consistency.py <폴더> [--extrinsics a.yaml b.yaml ...] [--min-corners 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_plus_plus_core.head_scan import board_consistency as bc, head_fk  # noqa: E402

CANDIDATES = {
    "현행(v2)": head_fk.EXTRINSICS_YAML,
    "v3": head_fk.CONFIG_DIR / "head_extrinsics_bolt_v3_candidate.yaml",
    "v4": head_fk.CONFIG_DIR / "head_extrinsics_joint_v4_candidate.yaml",
}


def load_detections(folder: Path, min_corners: int) -> list[dict]:
    """자세마다 프레임들의 코너 픽셀을 id 별로 평균한다. 코너가 min_corners 미만이면 그 자세는 뺀다."""
    out = []
    for rec in json.loads((folder / "poses.json").read_text(encoding="utf-8"))["records"]:
        acc: dict[int, list[np.ndarray]] = {}
        k_mat = None
        for k in range(rec["frames"]):
            data = np.load(folder / f"pose_{rec['index']}_{k}.npz")
            k_mat = data["K"]
            found = bc.detect_corners(data["rgb"])
            if found:
                for i, uv in zip(*found):
                    acc.setdefault(int(i), []).append(uv)
        ids = np.array(sorted(acc))
        if len(ids) < min_corners:
            print(f"  자세 {rec['index']} {rec['commanded']}: 코너 {len(ids)}개 — 제외")
            continue
        out.append({"rec": rec, "k": k_mat, "ids": ids, "uv": np.array([np.mean(acc[int(i)], axis=0) for i in ids])})
        print(f"  자세 {rec['index']} {rec['commanded']} 실측 {rec['measured']}: 코너 {len(ids)}개")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--extrinsics", nargs="*", type=Path, default=None)
    ap.add_argument("--min-corners", type=int, default=6)
    args = ap.parse_args()
    cands = {p.stem: p for p in args.extrinsics} if args.extrinsics else {k: v for k, v in CANDIDATES.items() if v.exists()}
    dets = load_detections(args.folder, args.min_corners)
    if len(dets) < 2:
        print("❌ 코너가 충분한 자세가 2개 미만", file=sys.stderr)
        return 1
    for name, yaml_path in cands.items():
        obs = [bc.PoseObs(head_fk.t_base_cam(*d["rec"]["measured"], yaml_path), d["k"], d["ids"], d["uv"]) for d in dets]
        report = bc.consistency(obs)
        at = {z: report.rms_by_z[round(z, 4)] for z in (0.200, 0.205)}
        print(f"== {name}: 공통 코너 {report.n_common} · 자세 간 RMS 최소 {1000 * report.rms_at_best_m:.2f} mm (보드 높이 {report.best_z:.4f}) · "
              f"z 0.200 → {1000 * at[0.200]:.2f} mm · z 0.205 → {1000 * at[0.205]:.2f} mm")
        for d, off in zip(dets, report.pose_offset_m):
            scale, resid = bc.grid_scale(bc.PoseObs(head_fk.t_base_cam(*d["rec"]["measured"], yaml_path), d["k"], d["ids"], d["uv"]), report.best_z)
            print(f"   자세 {d['rec']['commanded']}: 합의 대비 오프셋 ({1000 * off[0]:+.2f}, {1000 * off[1]:+.2f}) mm · 격자 배율 {scale:.4f} · 격자 잔차 {1000 * resid:.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
