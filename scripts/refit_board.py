#!/usr/bin/env python3
"""ChArUco 보드 영상으로 목 FK 모델 항을 하나씩 더해 다시 맞추고, 자세 하나씩 빼고 검증한 오차(mm)를 비교한다. 목은 안 움직인다.

    python3 scripts/refit_board.py <capture_poses 폴더> [--yaml hand-eye.yaml] [--board-z 0.205]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_plus_plus_core.head_scan import head_fk, refit  # noqa: E402
from perception_plus_plus_core.head_scan.board_data import load_detections  # noqa: E402

VARIANTS = [("보정 없음(보드 위치만 맞춤)", ()), ("+ hand-eye 회전·이동", ("rot", "trans")),
            ("+ pan·tilt 영점", ("rot", "trans", "angles")), ("+ 초점거리 배율", ("rot", "trans", "angles", "focal")),
            ("hand-eye + 렌즈 왜곡 k1·k2", ("rot", "trans", "dist")),
            ("전부(왜곡 포함)", ("rot", "trans", "angles", "focal", "dist")),
            ("hand-eye + 보드별 높이", ("rot", "trans", "boardz")), ("hand-eye + 초점 + 보드별 높이", ("rot", "trans", "focal", "boardz"))]


def board_z_of(board, base_z: float) -> float:
    return base_z + board.dz


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--yaml", type=Path, default=head_fk.CONFIG_DIR / "head_extrinsics_joint_v4_candidate.yaml")
    ap.add_argument("--board-z", type=float, default=0.205)
    ap.add_argument("--min-corners", type=int, default=6)
    args = ap.parse_args()
    dets = [refit.Detection(pan=d["rec"]["measured"][0], tilt=d["rec"]["measured"][1], board=d["board"], ids=d["ids"], uv=d["uv"], k=d["k"])
            for d in load_detections(args.folder, args.min_corners, args.yaml)]
    print(f"기준 hand-eye {args.yaml.name} · 보드 높이 {args.board_z} · 관측 {len(dets)}개")
    for label, free in VARIANTS:
        fit = refit.fit(dets, args.yaml, args.board_z, free)
        lopo = refit.leave_one_pose_out(dets, args.yaml, args.board_z, free)
        p = fit.params
        heights = " ".join(f"{n}:{board_z_of(b, args.board_z):.4f}" for n, b in fit.boards.items())
        print(f"[{label}] 맞춤 잔차 {fit.rms_px:.3f} px · 자세 하나씩 빼고 검증 {lopo.rms_mm:.2f} mm"
              f" · 회전 {tuple(round(float(v) * 1000, 2) for v in p.rot)} mrad · 이동 {tuple(round(float(v) * 1000, 2) for v in p.trans)} mm"
              f" · pan/tilt 영점 {p.pan_off_deg:+.3f}/{p.tilt_off_deg:+.3f}° · 초점 배율 {p.f_scale:.4f} · 왜곡 k1 {p.k1:+.4f} k2 {p.k2:+.4f} · 보드 높이 {heights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
