"""목 자세 서베이의 순수 부분 — 자세 목록 검증, `head_home.py` 출력에서 실측 각도 읽기.

`head_home.py` 는 정착 4 s 뒤 30샘플 평균을 `실제 +15.136°` 형태로 찍는다. 서베이는 그 평균을 **실측 각도**로 쓴다
(명령값이 아님). 평균 형식이 바뀌면 파싱이 실패해 서베이가 멈추므로 틀린 각도로 좌표를 내지 않는다.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

#: 사용자 지정(2026-10-13): tilt 는 중력 방향(0 쪽)으로 기준 -20 에서 10° 까지만. 위쪽(음수 쪽)과 pan 은 제한 없음.
DEFAULT_MAX_TILT_DEG = -10.0

_LINE = re.compile(r"^\s*(pan|tilt):\s*목표\s*([+-][\d.]+)°\s*·\s*실제\s*([+-][\d.]+)°.*$")


@dataclass(frozen=True)
class HeadReading:
    """head_home 검증 결과: 실측 평균 각도(인코더 deg)와 두 모터 모두 통과 여부."""

    pan_deg: float
    tilt_deg: float
    ok: bool


def parse_head_home(text: str) -> HeadReading:
    """`head_home.py --execute` 출력에서 pan·tilt 실측 평균과 통과 여부(✓)를 읽는다."""
    measured: dict[str, float] = {}
    ok = True
    for line in text.splitlines():
        match = _LINE.match(line)
        if match:
            measured[match.group(1)] = float(match.group(3))
            ok &= "✓" in line
    if set(measured) != {"pan", "tilt"}:
        raise ValueError("head_home 출력에서 pan/tilt 실측값을 못 읽었다")
    return HeadReading(pan_deg=measured["pan"], tilt_deg=measured["tilt"], ok=ok)


def parse_pose_list(text: str) -> list[tuple[float, float]]:
    """`15,-20;-15,-20` → [(15.0, -20.0), (-15.0, -20.0)]."""
    poses = []
    for item in text.split(";"):
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"자세는 'pan,tilt' 형식이어야 한다: {item!r}")
        poses.append((float(parts[0]), float(parts[1])))
    return poses


def validate_poses(poses: list[tuple[float, float]], max_tilt_deg: float = DEFAULT_MAX_TILT_DEG) -> None:
    """모든 자세가 유한하고 tilt 가 max_tilt_deg 이하인지 — 하나라도 어긋나면 실행 전에 거부한다."""
    if not poses:
        raise ValueError("자세가 없다")
    for pan, tilt in poses:
        if not (math.isfinite(pan) and math.isfinite(tilt)):
            raise ValueError(f"유한하지 않은 각도: {(pan, tilt)}")
        if tilt > max_tilt_deg:
            raise ValueError(f"tilt {tilt}° 는 중력 방향 한계 {max_tilt_deg}° 를 넘는다")
