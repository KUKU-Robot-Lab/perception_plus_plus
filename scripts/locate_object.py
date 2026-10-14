#!/usr/bin/env python3
"""물체 하나를 찾아 base 좌표를 출력한다: 목 이동 → 촬영 → 목 실측 각도 FK → z 고정 CAD 실루엣 맞춤 → 기준 자세 복귀.

    python3 scripts/locate_object.py blue_cup --side right --dry-run     # 계획만 출력(목 안 움직임)
    python3 scripts/locate_object.py blue_cup --side right --execute     # 목이 움직인다 — 사용자 허락 뒤에만

--execute 는 목을 움직인다. 허락은 그 실행 한 번에만 유효하다. 영상 스트림이 vision 호스트에서 돌고 있어야 한다
(--start-perception 이면 survey_head_poses.py 와 같은 방식으로 런처와 perception_ctl 을 먼저 띄운다).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_plus_plus_core.head_scan import adapters, calib_profile, locate, objects, survey  # noqa: E402


def start_perception(sim2real: Path, vision_host: str, names: list[str]) -> subprocess.Popen:
    launcher = subprocess.Popen(
        ["python3", str(sim2real / "scripts/nodes/perception_launcher_node.py"), "--host", vision_host, "--pose-dest", "169.254.186.1",
         "--camera-extrinsics", str(sim2real / "config/global_camera_extrinsics.yaml")],
        cwd=sim2real, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import time

    time.sleep(4)
    ctl = adapters.run(["python3", str(sim2real / "scripts/ops/perception_ctl.py"), "start", *names, "--wait", "150"], cwd=sim2real, timeout=220)
    if ctl.returncode != 0:
        launcher.terminate()
        raise RuntimeError(f"perception_ctl start 실패: {ctl.stdout[-300:]}")
    return launcher


def run_fake(args, spec, request) -> int:
    """가짜 목·카메라로 전체 흐름을 돌린다(로봇·카메라·ssh 없음)."""
    import cv2
    import numpy as np

    from perception_plus_plus_core.head_scan import fakes

    truth = tuple(float(v) for v in args.fake.split(","))
    color = tuple(int(c) for c in cv2.cvtColor(np.array([[[int(spec.hue), 255, 255]]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0])
    head = fakes.FakeHead()
    result = locate.locate_object(request, spec, head, fakes.FakeCamera(head, truth, spec, color))
    print(f"[fake] 이동 {head.moves} · 복귀 {head.homed}회")
    print(f"[fake] 찾음={result.found}" + (f" 좌표 ({result.x:.4f}, {result.y:.4f}, {result.z:.4f}) 정답 {truth}" if result.found else f" {result.reason}"))
    return 0 if result.found else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("object", help=f"물체 이름 {sorted(objects.OBJECTS)} 또는 별칭 {sorted(objects.ALIASES)}")
    ap.add_argument("--side", choices=["right", "left"], default=None, help="찾을 쪽(로봇 기준). 없으면 전체")
    ap.add_argument("--poses", default=None, help="'pan,tilt;pan,tilt' 를 주면 계획 대신 그 자세만 쓴다")
    ap.add_argument("--max-poses", type=int, default=locate.DEFAULT_MAX_POSES)
    ap.add_argument("--min-used", type=int, default=locate.DEFAULT_MIN_USED)
    ap.add_argument("--frames", type=int, default=1, help="자세당 촬영 장수")
    ap.add_argument("--sim2real", type=Path, default=adapters.SIM2REAL)
    ap.add_argument("--vision-host", default="vision-3090")
    ap.add_argument("--extrinsics", type=Path, default=None, help="hand-eye yaml(기본: 저장소 head_extrinsics.yaml). 후보 비교용")
    ap.add_argument("--calib-profile", default=None, help="보정 프로필(이름 또는 경로, sim2real config/calib_profiles). hand-eye 와 목 각도별 보정을 쓰고, 계획도 프로필 각도로만 한다")
    ap.add_argument("--save-dir", type=Path, default=None, help="찍은 영상을 이 폴더에 frame_NN.npz 로 남긴다(진단용)")
    ap.add_argument("--start-perception", action="store_true")
    ap.add_argument("--fake", metavar="X,Y", help="로봇 없이 가짜 목·카메라로 돌린다(물체가 (X,Y) 에 있다고 치고 렌더). 목은 안 움직인다")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    spec = objects.load_spec(args.object)
    profile = calib_profile.load_profile(args.calib_profile) if args.calib_profile else None
    poses = survey.parse_pose_list(args.poses) if args.poses else None
    request = locate.LocateRequest(object_name=spec.name, side=args.side, poses=poses, max_poses=args.max_poses,
                                   min_used=args.min_used, n_frames=args.frames)
    plan = poses or (locate.plan_poses_from_profile(profile, args.side) if profile else locate.plan_poses(args.side, request.tilt_deg))[:args.max_poses]
    survey.validate_poses(plan + [locate.HOME_POSE])
    print(f"물체 {spec.name} · 쪽 {args.side} · 후보 자세 {plan} · 자세당 {args.frames}장 · 끝나면 {locate.HOME_POSE} 로 복귀")
    if args.fake:
        return run_fake(args, spec, request)
    if args.dry_run or not args.execute:
        busy = adapters.busy_head_commanders()
        print(f"[dry-run] 목을 명령하는 다른 프로세스 {len(busy)}개" + "".join(f"\n  {b}" for b in busy))
        print("[dry-run] 목을 움직이려면 --execute (사용자 허락 뒤)")
        return 0
    busy = adapters.busy_head_commanders()
    if busy:
        print("❌ 목을 명령하는 다른 프로세스가 있다 — 중단:\n  " + "\n  ".join(busy), file=sys.stderr)
        return 2
    launcher = start_perception(args.sim2real, args.vision_host, [spec.name]) if args.start_perception else None
    try:
        result = locate.locate_object(request, spec, adapters.Sim2RealHead(args.sim2real), adapters.SshCamera(args.vision_host, save_dir=args.save_dir),
                                      **({"extrinsics_yaml": args.extrinsics} if args.extrinsics else {}), profile=profile)
    finally:
        if launcher:
            launcher.terminate()
    for shot in result.shots:
        print(f"  촬영 pan {shot.pan_deg:+.3f} tilt {shot.tilt_deg:+.3f} 영상 {shot.stamp_s:.3f} 색 덩어리 {shot.n_blobs}" + (f" · 시작점 ({shot.init_xy[0]:.3f}, {shot.init_xy[1]:.3f})" if shot.init_xy else ""))
    print(f"기준 자세 복귀: {'✓' if result.home_ok else '❌ 확인 필요'}")
    if not result.found:
        print(f"못 찾음: {result.reason}")
        return 1
    print(f"{spec.name} 좌표 (x, y, z) = ({result.x:.4f}, {result.y:.4f}, {result.z:.4f}) m · 사용 {result.n_used} 제외 {result.n_rejected} · "
          f"흩어짐 {1000 * result.spread_m:.1f} mm · IoU {result.mean_iou:.2f}"
          + ("" if result.consistent else " · ⚠ 관측이 부족해 일관성 미확인"))
    return 0 if result.home_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
