# perception_plus_plus_core/tracking/anchor_geometry.py
"""ROS 비의존 anchor 기하 헬퍼 — 새 anchored 노드의 프레임별 판정 로직.

live_fp_demo.py에서 유래(검증됨). 기존 tracking 로직은 건드리지 않는다.
"""
from __future__ import annotations

import numpy as np


def reproject(pose: np.ndarray, K: np.ndarray) -> tuple[float, float] | None:
    p = pose[:3, 3]
    if p[2] <= 1e-6:
        return None
    return (float(K[0, 0] * p[0] / p[2] + K[0, 2]),
            float(K[1, 1] * p[1] / p[2] + K[1, 2]))


def inside_box(uv: tuple[float, float], xyxy, margin: float) -> bool:
    x0, y0, x1, y1 = xyxy
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return (x0 - mx <= uv[0] <= x1 + mx) and (y0 - my <= uv[1] <= y1 + my)


def anchor_valid(pose: np.ndarray, K: np.ndarray, det_xyxy,
                 z_min: float, z_max: float, margin: float) -> bool:
    """z 범위 + (검출 있으면) bbox 재투영 포함 여부. 검출 없으면 z만 본다."""
    z = float(pose[2, 3])
    if z < z_min or z > z_max:
        return False
    if det_xyxy is None:
        return True
    uv = reproject(pose, K)
    if uv is None:
        return False
    return inside_box(uv, det_xyxy, margin)


def bbox_depth_mask(depth: np.ndarray, xyxy, band: float = 0.08,
                    lo: float = 0.1, hi: float = 3.0) -> np.ndarray:
    """bbox ∩ 유효깊이의 near-percentile35 ± band. 2000px 미만이면 유효깊이 전체."""
    h, w = depth.shape
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    box = np.zeros((h, w), bool)
    box[y0:y1, x0:x1] = True
    valid = box & (depth > lo) & (depth < hi)
    vals = depth[valid]
    if vals.size == 0:
        return box
    centre = float(np.percentile(vals, 35))
    mask = valid & (depth > centre - band) & (depth < centre + band)
    return mask if mask.sum() >= 2000 else valid
