"""테이블 위에 선 물체의 base 좌표를 여러 목 자세의 영상에서 구한다(z 는 아는 값으로 고정) — 순수 계산.

자세마다 촬영 순간의 목 각도 FK 로 카메라 자세를 만들고, `silhouette_fit.fit_xy` 로 CAD 실루엣이 색 마스크와 맞는 (x, y)를
구한 뒤, 일치도(IoU)가 낮은 관측과 중앙값에서 먼 이상치를 버리고 IoU 가중 평균을 낸다. z = 상판 + 원점 높이, roll·pitch = 0.
FPP 포즈는 시작점으로만 쓴다. 한계: 마스크가 가려지거나 같은 색 배경이 있으면 그 관측은 버려진다(IoU 로 거름).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import head_fk, silhouette_fit

TABLE_TOP_Z = 0.205
DEFAULT_MIN_IOU = 0.85
DEFAULT_REJECT_M = 0.010
MIN_MASK_PX = 200


@dataclass(frozen=True)
class StandingObs:
    """한 번의 촬영: RGB, 카메라 내부 행렬, 촬영 순간의 목 인코더 각(deg), FPP→base 로 얻은 (x, y) 시작값."""

    rgb: np.ndarray
    k: np.ndarray
    pan_deg: float
    tilt_deg: float
    xy_init: tuple[float, float]


@dataclass(frozen=True)
class StandingEstimate:
    """결과 위치(base, m)와 일관성 지표. spread_m 는 채택 관측 (x, y) 의 평균에서 최대 거리."""

    x: float
    y: float
    z: float
    n_used: int
    n_rejected: int
    spread_m: float
    mean_iou: float


def estimate_standing(observations: list[StandingObs], vertices_body: np.ndarray, faces: np.ndarray,
                      origin_above_bottom: float, hue: float, extrinsics_yaml: Path | str = head_fk.EXTRINSICS_YAML,
                      min_iou: float = DEFAULT_MIN_IOU, reject_m: float = DEFAULT_REJECT_M,
                      base_z: float = TABLE_TOP_Z,
                      occluder: tuple[np.ndarray, np.ndarray, float] | None = None) -> StandingEstimate | None:
    """관측들에서 (x, y) 를 구한다. 쓸 만한 관측이 없으면 None(값을 만들어 내지 않는다).

    base_z 는 물체 바닥이 닿는 높이(기본 상판, 홀더 안이면 홀더 안쪽 바닥), occluder 는 `silhouette_fit.fit_xy` 참고.
    """
    z_origin = base_z + origin_above_bottom
    fits = []
    for obs in observations:
        t_bc = head_fk.t_base_cam(obs.pan_deg, obs.tilt_deg, extrinsics_yaml)
        uv, _ = silhouette_fit.project_points(np.array([[obs.xy_init[0], obs.xy_init[1], z_origin]]), t_bc, obs.k)
        mask = silhouette_fit.largest_blob_near(silhouette_fit.hue_mask(obs.rgb, hue), tuple(uv[0]))
        if mask.sum() < MIN_MASK_PX:
            continue
        fit = silhouette_fit.fit_xy(vertices_body, faces, z_origin, t_bc, obs.k, mask, obs.xy_init, occluder=occluder)
        if fit.iou >= min_iou:
            fits.append(fit)
    if not fits:
        return None
    xy = np.array([[f.x, f.y] for f in fits])
    keep = np.ones(len(fits), dtype=bool)
    if len(fits) >= 3:
        keep = np.linalg.norm(xy - np.median(xy, axis=0), axis=1) <= reject_m
        if not keep.any():
            keep[:] = True
    weights = np.array([f.iou for f in fits])[keep]
    mean_xy = (xy[keep] * weights[:, None]).sum(axis=0) / weights.sum()
    spread = float(np.linalg.norm(xy[keep] - xy[keep].mean(axis=0), axis=1).max())
    return StandingEstimate(x=float(mean_xy[0]), y=float(mean_xy[1]), z=z_origin, n_used=int(keep.sum()),
                            n_rejected=int((~keep).sum()), spread_m=spread, mean_iou=float(np.mean(weights)))
