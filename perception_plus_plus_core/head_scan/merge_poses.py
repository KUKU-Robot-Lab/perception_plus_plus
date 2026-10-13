"""여러 목 자세에서 본 같은 물체의 카메라 프레임 포즈를 base 좌표로 합친다.

각 관측은 **촬영 순간의 목 인코더 각**과 함께 기록한다(목이 정지한 뒤 읽은 값). 수신 시각의 목 각도를 쓰는
기존 `object_pose_node` 와 달리, 관측마다 자기 각도로 변환하므로 목을 움직이며 스캔해도 좌표가 맞는다.

이상치는 위치 중앙값에서 `reject_m` 넘게 벗어난 관측을 버린다(관측 3개 이상일 때만). 회전은 쿼터니언
부호를 맞춘 평균이다. `table_z` 를 주면 그 물체의 z 만 테이블 제약값으로 덮어쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import head_fk

DEFAULT_REJECT_M = 0.03
MIN_OBS_FOR_REJECTION = 3


@dataclass(frozen=True)
class Observation:
    """한 번의 촬영: 물체 이름, 촬영 순간의 목 인코더 각(deg), 카메라 optical 프레임 기준 물체 포즈(4x4)."""

    name: str
    pan_deg: float
    tilt_deg: float
    t_cam_obj: np.ndarray
    weight: float = 1.0


@dataclass(frozen=True)
class MergedPose:
    """통합 결과. pos_spread_m 는 채택된 관측 위치의 평균에서 최대 거리(일관성 지표)."""

    name: str
    position: np.ndarray
    quat_wxyz: np.ndarray
    n_used: int
    n_rejected: int
    pos_spread_m: float


def _merge_one(name: str, obs: list[Observation], reject_m: float, table_z: float | None) -> MergedPose:
    poses = [head_fk.t_base_cam(o.pan_deg, o.tilt_deg) @ np.asarray(o.t_cam_obj, dtype=float) for o in obs]
    positions = np.array([p[:3, 3] for p in poses])
    keep = np.ones(len(obs), dtype=bool)
    if len(obs) >= MIN_OBS_FOR_REJECTION:
        keep = np.linalg.norm(positions - np.median(positions, axis=0), axis=1) <= reject_m
        if not keep.any():
            keep[:] = True
    weights = np.array([o.weight for o in obs], dtype=float)[keep]
    weights = weights / weights.sum()
    mean_pos = (positions[keep] * weights[:, None]).sum(axis=0)
    quats = np.array([head_fk.quat_wxyz_from_matrix(p[:3, :3]) for p in poses])[keep]
    quats = quats * np.where(quats @ quats[0] < 0, -1.0, 1.0)[:, None]
    mean_quat = (quats * weights[:, None]).sum(axis=0)
    mean_quat = mean_quat / np.linalg.norm(mean_quat)
    if table_z is not None:
        mean_pos = np.array([mean_pos[0], mean_pos[1], float(table_z)])
    spread = float(np.linalg.norm(positions[keep] - positions[keep].mean(axis=0), axis=1).max())
    return MergedPose(name=name, position=mean_pos, quat_wxyz=mean_quat, n_used=int(keep.sum()),
                      n_rejected=int((~keep).sum()), pos_spread_m=spread)


def merge(observations: list[Observation], reject_m: float = DEFAULT_REJECT_M,
          table_z: dict[str, float] | None = None) -> dict[str, MergedPose]:
    """물체 이름별로 관측을 합친다. 관측이 없으면 빈 dict."""
    grouped: dict[str, list[Observation]] = {}
    for o in observations:
        grouped.setdefault(o.name, []).append(o)
    table_z = table_z or {}
    return {name: _merge_one(name, obs, reject_m, table_z.get(name)) for name, obs in grouped.items()}
