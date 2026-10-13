#!/usr/bin/env python3
"""테이블 전체를 덮는 목 스캔 자세 세트를 계산해 출력한다(순수 계산, 로봇·GPU 불필요).

    python3 scripts/plan_head_scan.py                                   # 기본 범위
    python3 scripts/plan_head_scan.py --pan -90 90 --tilt -40 -10 --target 0.99 --json out.json

★기본 범위는 가정값이다. 실기 목 제한(사용자가 의도적으로 둔 값)을 확인한 뒤 --pan/--tilt 로 넘길 것.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception_plus_plus_core.head_scan import scan_poses  # noqa: E402

DEFAULT_PAN = (-75.0, 75.0)
DEFAULT_TILT = (-40.0, -10.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pan", type=float, nargs=2, default=DEFAULT_PAN, metavar=("MIN", "MAX"))
    ap.add_argument("--tilt", type=float, nargs=2, default=DEFAULT_TILT, metavar=("MIN", "MAX"))
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    points = scan_poses.table_points()
    result = scan_poses.greedy_scan_set(points, tuple(args.pan), tuple(args.tilt), args.step, args.target)
    print(f"범위 pan {tuple(args.pan)} tilt {tuple(args.tilt)} step {args.step} → "
          f"{len(result.poses)}자세, 커버 {result.covered_fraction:.1%} (목표 {args.target:.0%})")
    for (pan, tilt), gain in zip(result.poses, result.per_pose_gain):
        print(f"  pan {pan:+6.1f}  tilt {tilt:+6.1f}  새로 덮는 점 {gain}")
    if result.covered_fraction < args.target:
        print("목표에 못 미친다 — 범위를 넓히거나 목표를 낮출 것", file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps({"poses": result.poses, "covered_fraction": result.covered_fraction},
                                        ensure_ascii=False), encoding="utf-8")
    return 0 if result.covered_fraction >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
