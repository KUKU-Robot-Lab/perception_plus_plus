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

_LINE = re.compile(r"^\s*(pan|tilt):\s*목표\s*([+-][\d.]+)°\s*·\s*실제\s*([+-][\d.]+)°\s*·\s*오차\s*([+-][\d.]+)°\s*·\s*σ\s*([\d.]+)°.*$")
#: 정착 판정(사용자 방침: 명령과 실제 각도는 다를 수 있으니 실측을 쓴다). 정지해 있고 명령에 가까우면 받아들인다.
MAX_ERR_DEG = 3.0
MAX_SIGMA_DEG = 0.1


@dataclass(frozen=True)
class HeadReading:
    """head_home 검증 결과: 실측 평균 각도(인코더 deg)와 두 모터 모두 통과 여부."""

    pan_deg: float
    tilt_deg: float
    ok: bool
    pan_err_deg: float = 0.0
    tilt_err_deg: float = 0.0
    pan_sigma_deg: float = 0.0
    tilt_sigma_deg: float = 0.0


def parse_head_home(text: str) -> HeadReading:
    """`head_home.py --execute` 출력에서 pan·tilt 실측 평균과 통과 여부(✓)를 읽는다."""
    rows: dict[str, tuple[float, float, float]] = {}
    ok = True
    for line in text.splitlines():
        match = _LINE.match(line)
        if match:
            rows[match.group(1)] = (float(match.group(3)), float(match.group(4)), float(match.group(5)))
            ok &= "✓" in line
    if set(rows) != {"pan", "tilt"}:
        raise ValueError("head_home 출력에서 pan/tilt 실측값을 못 읽었다")
    return HeadReading(pan_deg=rows["pan"][0], tilt_deg=rows["tilt"][0], ok=ok,
                       pan_err_deg=rows["pan"][1], tilt_err_deg=rows["tilt"][1],
                       pan_sigma_deg=rows["pan"][2], tilt_sigma_deg=rows["tilt"][2])


def accept(reading: HeadReading, max_err_deg: float = MAX_ERR_DEG, max_sigma_deg: float = MAX_SIGMA_DEG) -> bool:
    """검증(✓)이 실패해도 정지해 있고(σ 작음) 명령에서 크게 안 벗어났으면 실측 각도로 계속한다."""
    return (max(abs(reading.pan_err_deg), abs(reading.tilt_err_deg)) <= max_err_deg
            and max(reading.pan_sigma_deg, reading.tilt_sigma_deg) <= max_sigma_deg)


def drop_stale(records: list[dict], min_move_deg: float = 0.5) -> list[dict]:
    """목이 움직였는데(min_move_deg 초과) 이미 본 값과 위치가 완전히 같은 원시 포즈는 새로 찍히지 않은 낡은 값이라 버린다.

    원본은 바꾸지 않는다. 비교 기준은 같은 물체에서 지금까지 남긴 모든 기록이다(직전 것만이 아니다).
    """
    kept: list[dict] = []
    seen: dict[str, list[tuple[list[float], list[float]]]] = {}  # 물체 → [(그때의 원시 위치, 그때의 실측 각도)]
    for rec in records:
        fresh = {}
        for name, raw in rec.get("raw_cam", {}).items():
            stale = any(pos == raw["pos"] and max(abs(a - b) for a, b in zip(rec["measured"], pose)) > min_move_deg
                        for pos, pose in seen.get(name, []))
            if stale:
                continue
            fresh[name] = raw
            seen.setdefault(name, []).append((raw["pos"], rec["measured"]))
        kept.append({**rec, "raw_cam": fresh})
    return kept


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
