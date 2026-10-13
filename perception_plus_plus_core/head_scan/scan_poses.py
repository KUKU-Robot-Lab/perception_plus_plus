"""테이블 전체를 덮는 목 스캔 자세 후보 계산 — 순수 계산(로봇·GPU 없음).

테이블 상면 격자점을 목 FK 로 구한 카메라 자세에 투영해, 이미지 안에 들어오는 점 집합을 자세마다
구하고 탐욕적으로 최소 자세 세트를 고른다.

한계: 가시성은 팔·몸통·물체 가림을 반영하지 않는다. 카메라 내부 파라미터는 `head_camera_sim.json`
(sim 사양) 값이라 실기 보정값과 다를 수 있다. 거리 범위는 depth 기준 가정이라 색 검출만 쓸 때도 같이 거른다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import head_fk

#: sim2real `scripts/calib/table_cad_extrinsics.py` 의 테이블 상수(2026-10-13 확인).
TABLE_TOP_Z = 0.205
TABLE_X = (0.07, 0.47)
TABLE_Y = (-0.45, 0.45)

CAMERA_SPEC_JSON = head_fk.CONFIG_DIR / "head_camera_sim.json"
#: 이미지 가장자리에서 이만큼 안쪽에 들어와야 보인다고 친다(검출 안정성 여유).
DEFAULT_EDGE_MARGIN_PX = 40
#: ★가정값 — D435 계열 depth 최소 거리. 장치별 실측으로 교체할 것.
DEFAULT_MIN_RANGE_M = 0.30
DEFAULT_MAX_RANGE_M = 1.50


@dataclass(frozen=True)
class Camera:
    """핀홀 카메라 사양. 거리·가장자리 여유는 가시 판정용 가정값이다."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    edge_margin_px: float = DEFAULT_EDGE_MARGIN_PX
    min_range_m: float = DEFAULT_MIN_RANGE_M
    max_range_m: float = DEFAULT_MAX_RANGE_M


def default_camera(edge_margin_px: float = DEFAULT_EDGE_MARGIN_PX,
                   min_range_m: float = DEFAULT_MIN_RANGE_M,
                   max_range_m: float = DEFAULT_MAX_RANGE_M) -> Camera:
    spec = json.loads(Path(CAMERA_SPEC_JSON).read_text(encoding="utf-8"))
    k = spec["intrinsic_matrix"]
    return Camera(width=int(spec["width"]), height=int(spec["height"]),
                  fx=k[0], fy=k[4], cx=k[2], cy=k[5],
                  edge_margin_px=edge_margin_px, min_range_m=min_range_m, max_range_m=max_range_m)


def table_points(step_m: float = 0.02) -> np.ndarray:
    """테이블 상면 격자점(base 좌표, N x 3)."""
    xs = np.linspace(TABLE_X[0], TABLE_X[1], int(round((TABLE_X[1] - TABLE_X[0]) / step_m)) + 1)
    ys = np.linspace(TABLE_Y[0], TABLE_Y[1], int(round((TABLE_Y[1] - TABLE_Y[0]) / step_m)) + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, TABLE_TOP_Z)])


def project(points_base: np.ndarray, t_base_cam: np.ndarray, cam: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """base 좌표 점들을 카메라 이미지에 투영한다 → (u, v, 광축 방향 깊이 z). z <= 0 이면 u, v 는 무의미하다."""
    inv = np.linalg.inv(t_base_cam)
    cam_pts = np.asarray(points_base, dtype=float) @ inv[:3, :3].T + inv[:3, 3]
    z = cam_pts[:, 2]
    safe_z = np.where(z > 1e-9, z, 1.0)
    return cam.fx * cam_pts[:, 0] / safe_z + cam.cx, cam.fy * cam_pts[:, 1] / safe_z + cam.cy, z


def visible_mask(points_base: np.ndarray, t_base_cam: np.ndarray, cam: Camera) -> np.ndarray:
    """base 좌표 점들이 카메라 이미지 안(가장자리 여유, 거리 범위 포함)에 보이는지."""
    u, v, z = project(points_base, t_base_cam, cam)
    m = cam.edge_margin_px
    return ((z >= cam.min_range_m) & (z <= cam.max_range_m)
            & (u >= m) & (u <= cam.width - m) & (v >= m) & (v <= cam.height - m))


@dataclass(frozen=True)
class ScanSet:
    """탐욕 선택 결과. covered_fraction 은 대상 점 중 하나라도 본 비율, per_pose_gain 은 자세별 새로 덮은 점 수."""

    poses: tuple[tuple[float, float], ...]  # (pan_encoder_deg, tilt_encoder_deg)
    covered_fraction: float
    per_pose_gain: tuple[int, ...]


def _grid(rng: tuple[float, float], step: float) -> list[float]:
    """[lo, hi] 안에서만 step 간격 값을 만든다(hi 를 넘지 않는다)."""
    if step <= 0:
        raise ValueError(f"step 은 양수여야 한다: {step}")
    lo, hi = rng
    if hi < lo:
        raise ValueError(f"범위가 뒤집혔다: {rng}")
    n = int(math.floor((hi - lo) / step + 1e-9))
    return [lo + i * step for i in range(n + 1)]


def greedy_scan_set(points: np.ndarray, pan_range: tuple[float, float], tilt_range: tuple[float, float],
                    step_deg: float = 5.0, target: float = 0.99,
                    cam: Camera | None = None) -> ScanSet:
    """격자 후보 중 새로 덮는 점이 가장 많은 자세를 target 까지 반복해 고른다.

    각도는 실기 인코더 deg. 이동 비용·순서는 고려하지 않고, 같은 이득이면 앞선 후보(pan, tilt 오름차순)가 이긴다.
    """
    cam = cam or default_camera()
    candidates = [(pan, tilt) for pan in _grid(pan_range, step_deg) for tilt in _grid(tilt_range, step_deg)]
    masks = {c: visible_mask(points, head_fk.t_base_cam(*c), cam) for c in candidates}
    covered = np.zeros(len(points), dtype=bool)
    chosen: list[tuple[float, float]] = []
    gains: list[int] = []
    while covered.mean() < target:
        best, best_gain = None, 0
        for cand in candidates:
            if cand in chosen:
                continue
            gain = int((masks[cand] & ~covered).sum())
            if gain > best_gain:
                best, best_gain = cand, gain
        if best is None:
            break
        chosen.append(best)
        gains.append(best_gain)
        covered |= masks[best]
    return ScanSet(poses=tuple(chosen), covered_fraction=float(covered.mean()),
                   per_pose_gain=tuple(gains))
