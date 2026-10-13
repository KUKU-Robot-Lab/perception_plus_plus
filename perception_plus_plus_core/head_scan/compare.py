"""(A) 정적 외부 파라미터와 (B) 목 FK 로 같은 관측을 base 좌표로 바꿔 자세 간 흩어짐을 비교한다.

물체는 가만히 있으므로 어느 목 자세에서 찍어도 base 좌표가 같아야 한다. 흩어짐(평균에서 최대 거리)이 오차의
직접 척도다 — 정답 좌표가 없어도 된다. 입력의 t_cam_obj 는 카메라 optical 프레임 기준 물체 포즈(FP++ 출력).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import head_fk


@dataclass(frozen=True)
class Record:
    """한 번의 촬영: 물체 이름, 촬영 순간의 목 인코더 각(deg), 카메라 프레임 기준 물체 포즈(4x4)."""

    name: str
    pan_deg: float
    tilt_deg: float
    t_cam_obj: np.ndarray


@dataclass(frozen=True)
class Comparison:
    """물체 하나의 비교 결과. 모든 거리는 m. max_static_vs_fk_m 은 같은 관측에 대한 (A)와 (B)의 위치 차이 최대값."""

    n: int
    mean_static: np.ndarray
    mean_fk: np.ndarray
    spread_static_m: float
    spread_fk_m: float
    max_static_vs_fk_m: float


def _spread(points: np.ndarray) -> float:
    return float(np.linalg.norm(points - points.mean(axis=0), axis=1).max())


def compare(records: list[Record], t_base_cam_static: np.ndarray) -> dict[str, Comparison]:
    """물체별로 (A)=정적 T_base_cam, (B)=각 촬영 목 각도의 FK 를 적용한 위치의 흩어짐을 낸다."""
    grouped: dict[str, list[Record]] = {}
    for rec in records:
        grouped.setdefault(rec.name, []).append(rec)
    result: dict[str, Comparison] = {}
    for name, recs in grouped.items():
        pos_a = np.array([(t_base_cam_static @ r.t_cam_obj)[:3, 3] for r in recs])
        pos_b = np.array([(head_fk.t_base_cam(r.pan_deg, r.tilt_deg) @ r.t_cam_obj)[:3, 3] for r in recs])
        result[name] = Comparison(
            n=len(recs), mean_static=pos_a.mean(axis=0), mean_fk=pos_b.mean(axis=0),
            spread_static_m=_spread(pos_a), spread_fk_m=_spread(pos_b),
            max_static_vs_fk_m=float(np.linalg.norm(pos_a - pos_b, axis=1).max()))
    return result
