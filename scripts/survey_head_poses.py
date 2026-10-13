#!/usr/bin/env python3
"""목 자세 서베이(5090) — 자세마다 목을 옮기고, 실측 각도를 읽고, 컬러 영상 + FP++ 원시 포즈를 모은다.

    python3 scripts/survey_head_poses.py --dry-run                         # 계획만 출력(아무것도 켜지 않음)
    python3 scripts/survey_head_poses.py --out-dir <폴더> --execute        # 실제 실행(목이 움직인다)

★`--execute` 는 목을 움직인다 — sim2real 규칙대로 사용자 허락 뒤에만. 허락은 이 한 번의 실행 전체를 덮는다.
순서: 사전 점검(로봇 세션이 돌면 중단) → 런처·수신기 기동 → FP++ 켜기 → 자세마다 [head_home(실측 평균) → 영상 저장 →
fpp_rescan → 원시 포즈 수집] → **항상** 기준 자세로 복귀 → 띄운 프로세스 종료. 자세 검증이 실패하면 거기서 멈추고 복귀한다.
tilt 는 인코더 기준 `--max-tilt`(기본 -10°) 이하만 허용한다(중력 방향 한계). 컨테이너는 내리지 않는다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception_plus_plus_core.head_scan import survey  # noqa: E402

HOME_POSE = (0.0, -20.0)
DEFAULT_POSES = "15,-20;-15,-20;0,-20"
DEFAULT_OBJECTS = ("source240b_blue", "shaker_c_orange")
BUSY_MARKERS = ("episode_ctl", "rh_aglt_node", "pd_right", "pd_left", "policy_control")


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def busy_robot_processes() -> list[str]:
    out = run(["ps", "-eo", "pid,cmd"]).stdout.splitlines()
    # 상황판(s2r_console) UI·브리지는 노드 이름을 인자로 들고 있을 뿐 로봇을 움직이는 프로세스가 아니다.
    return [ln.strip()[:120] for ln in out
            if any(m in ln for m in BUSY_MARKERS) and "survey_head_poses" not in ln and "s2r_console" not in ln]


def head_home(sim2real: Path, pan: float | None, tilt: float | None) -> survey.HeadReading:
    cmd = [str(sim2real / ".venv/bin/python"), str(sim2real / "scripts/head_home.py"), "--execute"]
    if pan is not None:
        cmd += ["--pan", str(pan), "--tilt", str(tilt)]
    res = run(cmd, cwd=sim2real, timeout=90)
    if res.returncode != 0:
        raise RuntimeError(f"head_home 실패 rc={res.returncode}: {res.stderr[-300:]}")
    return survey.parse_head_home(res.stdout)


class RawPoseCollector:
    """fpp_pose_rx 가 내는 카메라 프레임 원시 포즈를 모은다. ★구독은 fpp_rescan **전에** 시작한다(발행이 일회성이라 뒤늦게 붙으면 놓친다)."""

    def __init__(self, sim2real: Path, objects):
        sys.path.insert(0, str(sim2real / "scripts"))
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from object_registry import input_topic
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()
        self.got: dict = {}
        self.since_s = 0.0
        self.node = Node("survey_collect")
        for n in objects:
            self.node.create_subscription(PoseStamped, input_topic(n), lambda m, n=n: self._on(n, m), 10)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _on(self, name, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp >= self.since_s:
            q, p = msg.pose.orientation, msg.pose.position
            self.got[name] = {"pos": [p.x, p.y, p.z], "quat_wxyz": [q.w, q.x, q.y, q.z], "stamp_s": stamp}

    def begin(self, since_s: float) -> None:
        """이 시각 이후 stamp 만 받는다(옛 값 배제)."""
        self.got = {}
        self.since_s = since_s

    def result(self) -> dict:
        return dict(self.got)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--poses", default=DEFAULT_POSES, help="'pan,tilt;pan,tilt' 인코더 deg")
    ap.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    ap.add_argument("--max-tilt", type=float, default=survey.DEFAULT_MAX_TILT_DEG)
    ap.add_argument("--sim2real", type=Path, default=Path.home() / "rl_ws/sim2real")
    ap.add_argument("--vision-host", default="vision-3090")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    poses = survey.parse_pose_list(args.poses)
    survey.validate_poses(poses + [HOME_POSE], args.max_tilt)
    print(f"자세 {poses} → 마지막에 기준 자세 {HOME_POSE} 로 복귀 · tilt 한계 {args.max_tilt}° · 물체 {args.objects}")
    if args.dry_run or not args.execute:
        busy = busy_robot_processes()
        print(f"[dry-run] 로봇 세션 프로세스 {len(busy)}개" + ("".join(f"\n  {b}" for b in busy) if busy else ""))
        print("[dry-run] 목을 움직이려면 --execute (사용자 허락 뒤)")
        return 0
    busy = busy_robot_processes()
    if busy:
        print("❌ 로봇 세션이 돌고 있다 — 중단:\n  " + "\n  ".join(busy), file=sys.stderr)
        return 2
    if args.out_dir is None:
        print("❌ --out-dir 필요", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return execute(args, poses)


def execute(args, poses) -> int:
    s2r = args.sim2real
    children: list[subprocess.Popen] = []
    records: list[dict] = []
    status = 0
    try:
        for script, extra in (("perception_launcher_node.py", ["--host", args.vision_host, "--pose-dest", "169.254.186.1",
                                                              "--camera-extrinsics", str(s2r / "config/global_camera_extrinsics.yaml")]),
                              ("fpp_pose_rx.py", []),
                              # fpp_rescan 은 /objects/*/pose(base 좌표)를 읽어야 끝난다 — 변환 노드를 같이 띄운다.
                              ("object_pose_node.py", ["--camera-extrinsics", str(s2r / "config/global_camera_extrinsics.yaml"),
                                                       "--objects", *args.objects])):
            children.append(subprocess.Popen(["python3", str(s2r / "scripts/nodes" / script), *extra], cwd=s2r,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        collector = RawPoseCollector(s2r, args.objects)
        time.sleep(4)
        ctl = run(["python3", str(s2r / "scripts/ops/perception_ctl.py"), "start", *args.objects, "--wait", "150"], cwd=s2r, timeout=220)
        if ctl.returncode != 0:
            raise RuntimeError(f"perception_ctl start 실패: {ctl.stdout[-300:]}")
        for i, (pan, tilt) in enumerate(poses):
            reading = head_home(s2r, pan, tilt)
            if not reading.ok:
                raise RuntimeError(f"자세 {i} 검증 실패(오차 한계 초과): 실측 {reading}")
            time.sleep(args.settle_s)
            frame = args.out_dir / f"frame_{i}.npz"
            remote = f"/tmp/survey_frame_{i}.npz"
            cap = run(["ssh", "-o", "BatchMode=yes", args.vision_host,
                       f"cd ~/rl_ws/sim2real && source scripts/vision/common.sh && cd ~/rl_ws/perception_plus_plus && python3 scripts/save_color_frame.py --out {remote}"], timeout=60)
            ok_frame = cap.returncode == 0 and run(["scp", "-q", f"{args.vision_host}:{remote}", str(frame)]).returncode == 0
            t_start = time.time()
            collector.begin(t_start - 1.0)
            resc = run(["python3", str(s2r / "scripts/ops/fpp_rescan.py"), *args.objects, "--wait", "60",
                        "--status-host", args.vision_host], cwd=s2r, timeout=150)
            time.sleep(1.0)
            raw = collector.result()
            records.append({"index": i, "commanded": [pan, tilt], "measured": [reading.pan_deg, reading.tilt_deg],
                            "frame": frame.name if ok_frame else None, "rescan_rc": resc.returncode, "raw_cam": raw})
            print(f"[{i}] 명령 {pan:+.1f}/{tilt:+.1f} 실측 {reading.pan_deg:+.3f}/{reading.tilt_deg:+.3f} · 영상 {'ok' if ok_frame else '없음'} · 찾은 물체 {sorted(raw)}")
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 복귀를 막지 않는다
        print(f"❌ 중단: {exc}", file=sys.stderr)
        status = 1
    finally:
        try:
            home = head_home(s2r, None, None)
            print(f"복귀 {HOME_POSE}: 실측 {home.pan_deg:+.3f}/{home.tilt_deg:+.3f} · {'✓' if home.ok else '❌ 확인 필요'}")
            status = status or (0 if home.ok else 3)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 복귀 실패 — 수동 확인 필요: {exc}", file=sys.stderr)
            status = 3
        for child in children:
            child.terminate()
        (args.out_dir / "survey.json").write_text(json.dumps({"records": records, "status": status}, ensure_ascii=False, indent=1), encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
