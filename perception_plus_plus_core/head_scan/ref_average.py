"""기준 자세 라이브 여러 장의 외부 파라미터를 평균하고, 쓸 만한지 판정한다 — 순수 계산(ROS·카메라 무관).

10.27 규칙(사용자): 기준 자세에서 라이브 3장을 각각 풀어 평균한다. 장끼리 카메라 위치 차가 `max_spread_mm` 를 넘거나
재투영 평균이 `max_res_px` 를 넘으면 쓰지 않는다. 머리 틱은 찍기 전후가 같아야 한다(`tick_stable`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MAX_SPREAD_MM = 5.0
DEFAULT_MAX_RES_PX = 8.0


@dataclass(frozen=True)
class Solution:
    """한 장에서 푼 카메라 외부 파라미터(base ← camera)."""

    position_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    res_px: float  # 그 장의 재투영 평균


@dataclass(frozen=True)
class Average:
    position_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    spread_mm: float
    res_px: float
    accepted: bool
    reasons: tuple[str, ...]


def _align_quats(quats: np.ndarray) -> np.ndarray:
    """q 와 -q 는 같은 회전이므로 첫 장과 부호를 맞춘다."""
    ref = quats[0]
    signs = np.where(quats @ ref < 0.0, -1.0, 1.0)
    return quats * signs[:, None]


def average(solutions: list[Solution], max_spread_mm: float = DEFAULT_MAX_SPREAD_MM,
            max_res_px: float = DEFAULT_MAX_RES_PX) -> Average:
    """장들을 평균하고 거부 사유를 모은다. 사유가 없으면 accepted."""
    if not solutions:
        raise ValueError("평균할 장이 없다")
    pos = np.array([s.position_m for s in solutions], dtype=float)
    quats = _align_quats(np.array([s.quat_wxyz for s in solutions], dtype=float))
    mean_pos = pos.mean(axis=0)
    mean_quat = quats.mean(axis=0)
    norm = np.linalg.norm(mean_quat)
    if norm < 1e-12:
        raise ValueError("회전 평균이 0 이다 — 장의 자세가 서로 반대일 수 있다")
    mean_quat = mean_quat / norm
    spread_mm = float(np.linalg.norm(pos - mean_pos, axis=1).max() * 1000.0)
    res_px = float(np.mean([s.res_px for s in solutions]))
    reasons: list[str] = []
    if spread_mm > max_spread_mm:
        reasons.append(f"장끼리 위치 차 {spread_mm:.1f} mm > {max_spread_mm:g}")
    if res_px > max_res_px:
        reasons.append(f"재투영 {res_px:.2f} px > {max_res_px:g}")
    return Average(
        position_m=(float(mean_pos[0]), float(mean_pos[1]), float(mean_pos[2])),
        quat_wxyz=(float(mean_quat[0]), float(mean_quat[1]), float(mean_quat[2]), float(mean_quat[3])),
        spread_mm=spread_mm,
        res_px=res_px,
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def tick_stable(before: dict[str, int], after: dict[str, int]) -> bool:
    """찍기 전후 머리 틱이 모두 같은가. 틱이 없으면 확인할 수 없으므로 False."""
    return bool(before) and before == after
