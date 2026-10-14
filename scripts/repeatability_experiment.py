#!/usr/bin/env python3
"""오차 분해 실험: 같은 자세에서 여러 프레임(촬영 잡음), 다른 자세로 갔다 돌아와 여러 번(목 재현성)을 나눠 잰다.

    python3 scripts/repeatability_experiment.py blue_cup --pose=-15,-20 --detour=15,-20 --dry-run
    python3 scripts/repeatability_experiment.py blue_cup --pose=-15,-20 --detour=15,-20 --execute   # 목이 움직인다: 사용자 허락 뒤

물체는 테이블 위(홀더 밖)에 두고, 실험 중 건드리지 않는다. 끝나면 기준 자세로 복귀한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_plus_plus_core.head_scan import adapters, locate, objects, repeatability, survey  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("object")
    ap.add_argument("--pose", required=True, help="'pan,tilt' 측정 자세")
    ap.add_argument("--detour", required=True, help="'pan,tilt' 갔다 돌아올 다른 자세")
    ap.add_argument("--visits", type=int, default=5)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--sim2real", type=Path, default=adapters.SIM2REAL)
    ap.add_argument("--vision-host", default="vision-3090")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    spec = objects.load_spec(args.object)
    (pose,), (detour,) = survey.parse_pose_list(args.pose), survey.parse_pose_list(args.detour)
    survey.validate_poses([pose, detour, locate.HOME_POSE])
    moves = args.visits * 2 - 1
    print(f"물체 {spec.name} · 측정 자세 {pose} · 우회 자세 {detour} · 방문 {args.visits}회 x {args.frames}장 · 목 이동 {moves}회 + 복귀")
    if args.dry_run or not args.execute:
        print("[dry-run] 목을 움직이려면 --execute (사용자 허락 뒤)")
        return 0
    busy = adapters.busy_head_commanders()
    if busy:
        print("❌ 목을 명령하는 다른 프로세스가 있다 — 중단:\n  " + "\n  ".join(busy), file=sys.stderr)
        return 2
    report = repeatability.run(adapters.Sim2RealHead(args.sim2real), adapters.SshCamera(args.vision_host), spec, pose, detour,
                               n_visits=args.visits, n_frames=args.frames)
    for i, v in enumerate(report.visits):
        print(f"  방문 {i}: 실측 pan {v.pan_deg:+.3f} tilt {v.tilt_deg:+.3f} · 맞춘 프레임 {len(v.fits)}/{args.frames}")
    print(f"촬영 잡음 {1000 * report.frame_noise_m:.2f} mm · 목 재현성(방문 평균 간 최대 거리) {1000 * report.visit_scatter_m:.2f} mm · "
          f"실측 pan 범위 {report.angle_scatter_deg:.3f}° · 복귀 {'✓' if report.home_ok else '❌ 확인 필요'}")
    return 0 if report.home_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
