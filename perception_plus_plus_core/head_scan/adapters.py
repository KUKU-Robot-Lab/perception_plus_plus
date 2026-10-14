"""실기 포트 구현 — `locate.HeadPort`·`CameraPort` 를 기존 도구(head_home.py, save_color_frame.py)로 채운다.

목: sim2real `scripts/head_home.py --execute` 를 부른다(토크 off, 모드, 게인, 목표, 토크 on 순서를 지키는 유일한 경로).
카메라: vision-3090 에서 `save_color_frame.py` 를 ssh 로 돌려 npz 를 받는다. 자세마다 몇 초 걸린다(v1).
이 모듈의 함수는 목을 움직이므로 호출 전에 사용자 허락이 있어야 한다(허락은 호출 한 번에만 유효).
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from . import locate, survey

SIM2REAL = Path.home() / "rl_ws/sim2real"
#: 목을 직접 명령하는 스크립트 파일명 — 같은 시리얼 포트를 쓰므로 동시에 돌면 안 된다. 팔 pd·손 노드는 목을 건드리지 않아 해당 없음.
# head_joint_publisher.py 는 목 각도를 읽어 발행만 한다(토크·게인·목표를 안 쓴다, 사용자 승인 2026-10-14) — 목록에서 제외.
HEAD_COMMANDER_SCRIPTS = frozenset({"head_home.py", "head_compliant_hold.py", "head_position_hold_node.py",
                                    "survey_head_poses.py", "locate_object.py"})


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def find_head_commanders(ps_lines: list[str], own_pid: int) -> list[str]:
    """`ps -eo pid,cmd` 줄들에서 목 명령 스크립트를 실행 중인 프로세스만 고른다(인자 안의 문자열 일치는 제외)."""
    found = []
    for line in ps_lines:
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit() or int(parts[0]) == own_pid:
            continue
        if any(Path(token).name in HEAD_COMMANDER_SCRIPTS for token in parts[1:4]):
            found.append(line.strip()[:120])
    return found


def busy_head_commanders() -> list[str]:
    """목을 명령하는 다른 프로세스 목록(ps 한 줄씩)."""
    return find_head_commanders(run(["ps", "-eo", "pid,cmd"]).stdout.splitlines()[1:], os.getpid())


class Sim2RealHead:
    """`head_home.py` 로 목을 움직이고 정착 뒤 30샘플 평균을 실측 각도로 돌려준다."""

    def __init__(self, sim2real: Path = SIM2REAL):
        self.sim2real = sim2real

    def _run(self, pan: float | None, tilt: float | None) -> survey.HeadReading:
        cmd = [str(self.sim2real / ".venv/bin/python"), str(self.sim2real / "scripts/head_home.py"), "--execute"]
        if pan is not None:
            cmd += ["--pan", str(pan), "--tilt", str(tilt)]
        res = run(cmd, cwd=self.sim2real, timeout=90)
        try:
            reading = survey.parse_head_home(res.stdout)
        except ValueError as exc:
            raise RuntimeError(f"head_home 출력을 못 읽음 rc={res.returncode}: {res.stdout[-300:]}{res.stderr[-300:]}") from exc
        if res.returncode != 0 and not survey.accept(reading):
            raise RuntimeError(f"head_home 정착 실패 rc={res.returncode}: 실측 {reading}")
        return reading

    def move(self, pan_deg: float, tilt_deg: float) -> survey.HeadReading:
        return self._run(pan_deg, tilt_deg)

    def go_home(self) -> survey.HeadReading:
        return self._run(None, None)


class SshCamera:
    """vision 호스트에서 컬러 영상을 한 장씩 받는다. 목 정지 직후 `settle_s` 만큼 기다린 뒤 찍는다(RGB 는 롤링 셔터)."""

    def __init__(self, host: str = "vision-3090", settle_s: float = 1.0, save_dir: Path | None = None):
        self.host, self.settle_s, self.save_dir = host, settle_s, save_dir
        self._count = 0

    def capture(self, n_frames: int = 1) -> list[locate.Frame]:
        time.sleep(self.settle_s)
        frames = []
        for _ in range(n_frames):
            remote = f"/tmp/locate_frame_{uuid.uuid4().hex[:8]}.npz"
            local = Path("/tmp") / Path(remote).name
            cap = run(["ssh", "-o", "BatchMode=yes", self.host,
                       f"cd ~/rl_ws/sim2real && source scripts/vision/common.sh && cd ~/rl_ws/perception_plus_plus && "
                       f"python3 scripts/save_color_frame.py --out {remote}"], timeout=60)
            if cap.returncode != 0 or run(["scp", "-q", f"{self.host}:{remote}", str(local)]).returncode != 0:
                raise RuntimeError(f"영상을 못 받았다: {cap.stderr[-200:]}")
            if self.save_dir is not None:  # 진단용 사본(찍은 순서대로 번호)
                self.save_dir.mkdir(parents=True, exist_ok=True)
                (self.save_dir / f"frame_{self._count:02d}.npz").write_bytes(local.read_bytes())
                self._count += 1
            data = np.load(local)
            frames.append(locate.Frame(rgb=data["rgb"], k=data["K"], stamp_s=float(data["stamp_s"])))
            local.unlink(missing_ok=True)
        return frames
