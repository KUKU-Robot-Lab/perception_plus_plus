#!/usr/bin/env python3
"""목 자세 목록을 차례로 가서 자세마다 N장 찍어 저장한다(실측 각도 포함). 끝나면 기준 자세로 복귀.

    python3 scripts/capture_poses.py --poses="-30,-25;0,-25;15,-25" --frames 3 --out-dir report/board_1014 --dry-run
    ... --execute        # 목이 움직인다: 사용자 허락 뒤에만(허락은 그 실행 한 번)

저장: pose_<i>_<k>.npz(rgb, K, stamp_s)와 poses.json(명령·실측 각도). 분석은 `board_consistency.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_plus_plus_core.head_scan import adapters, locate, survey  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--poses", required=True, help="'pan,tilt;pan,tilt' (인코더 deg). 음수가 있으면 --poses=\"...\" 로")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sim2real", type=Path, default=adapters.SIM2REAL)
    ap.add_argument("--vision-host", default="vision-3090")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    poses = survey.parse_pose_list(args.poses)
    survey.validate_poses(poses + [locate.HOME_POSE])
    print(f"자세 {poses} · 자세당 {args.frames}장 · 저장 {args.out_dir} · 끝나면 {locate.HOME_POSE} 로 복귀")
    if args.dry_run or not args.execute:
        print("[dry-run] 목을 움직이려면 --execute (사용자 허락 뒤)")
        return 0
    busy = adapters.busy_head_commanders()
    if busy:
        print("❌ 목을 명령하는 다른 프로세스가 있다 — 중단:\n  " + "\n  ".join(busy), file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    head, camera = adapters.Sim2RealHead(args.sim2real), adapters.SshCamera(args.vision_host)
    records, status = [], 0
    try:
        for i, (pan, tilt) in enumerate(poses):
            reading = head.move(pan, tilt)
            frames = camera.capture(args.frames)
            for k, frame in enumerate(frames):
                np.savez(args.out_dir / f"pose_{i}_{k}.npz", rgb=frame.rgb, K=frame.k, stamp_s=frame.stamp_s)
            records.append({"index": i, "commanded": [pan, tilt], "measured": [reading.pan_deg, reading.tilt_deg], "frames": len(frames)})
            print(f"[{i}] 명령 {pan:+.1f}/{tilt:+.1f} 실측 {reading.pan_deg:+.3f}/{reading.tilt_deg:+.3f} · {len(frames)}장")
    except Exception as exc:  # noqa: BLE001 - 실패해도 복귀한다
        print(f"❌ 중단: {exc}", file=sys.stderr)
        status = 1
    finally:
        try:
            home = head.go_home()
            print(f"복귀 {locate.HOME_POSE}: 실측 {home.pan_deg:+.3f}/{home.tilt_deg:+.3f} · {'✓' if home.ok else '❌ 확인 필요'}")
            status = status or (0 if home.ok else 3)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 복귀 실패 — 수동 확인 필요: {exc}", file=sys.stderr)
            status = 3
        (args.out_dir / "poses.json").write_text(json.dumps({"records": records, "status": status}, ensure_ascii=False, indent=1), encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
