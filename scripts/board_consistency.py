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


#: 같은 보드가 둘이다(2026-10-14). 마커 id 가 겹치므로 영상 속 덩어리를 대략적인 base y 로 가른다: 가운데 보드 A, -y 모서리 보드 B.
BOARD_B_MAX_Y = -0.20
ROUGH_Z = 0.200
B_ID_OFFSET = 100


def rough_board_label(t_base_cam: np.ndarray, k: np.ndarray, uv: np.ndarray) -> str:
    """덩어리 중심 픽셀을 z=보드 높이 평면에 얹은 base y 로 A/B 를 가른다(코너 id 구분용이라 대략이면 된다)."""
    obs = bc.PoseObs(t_base_cam, k, np.array([0]), uv.mean(axis=0, keepdims=True))
    return "B" if bc._plane_xy(obs, ROUGH_Z)[0, 1] < BOARD_B_MAX_Y else "A"


def load_detections(folder: Path, min_corners: int, label_yaml: Path) -> list[dict]:
    """자세마다 프레임들의 코너 픽셀을 (보드, id) 별로 평균한다. 코너가 min_corners 미만인 보드는 그 자세에서 뺀다."""
    out = []
    for rec in json.loads((folder / "poses.json").read_text(encoding="utf-8"))["records"]:
        acc: dict[tuple[str, int], list[np.ndarray]] = {}
        k_mat = None
        t_label = head_fk.t_base_cam(*rec["measured"], label_yaml)
        for k in range(rec["frames"]):
            data = np.load(folder / f"pose_{rec['index']}_{k}.npz")
            k_mat = data["K"]
            for ids, uv in bc.detect_boards(data["rgb"], min_corners=4):
                label = rough_board_label(t_label, k_mat, uv)
                for i, point in zip(ids, uv):
                    acc.setdefault((label, int(i)), []).append(point)
        for label in ("A", "B"):
            keys = sorted(key for key in acc if key[0] == label)
            if len(keys) < min_corners:
                if keys:
                    print(f"  자세 {rec['index']} {rec['commanded']} 보드 {label}: 코너 {len(keys)}개 — 제외")
                continue
            out.append({"rec": rec, "k": k_mat, "board": label, "ids": np.array([i for _, i in keys]),
                        "uv": np.array([np.mean(acc[key], axis=0) for key in keys])})
            print(f"  자세 {rec['index']} {rec['commanded']} 실측 {rec['measured']} 보드 {label}: 코너 {len(keys)}개")
    return out


def pose_obs(det: dict, yaml_path: Path, offset_b: bool = True) -> "bc.PoseObs":
    ids = det["ids"] + (B_ID_OFFSET if offset_b and det["board"] == "B" else 0)
    return bc.PoseObs(head_fk.t_base_cam(*det["rec"]["measured"], yaml_path), det["k"], ids, det["uv"])


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
