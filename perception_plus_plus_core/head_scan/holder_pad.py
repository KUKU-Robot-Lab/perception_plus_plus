"""컵홀더 흰 패드(30 mm, 마커 포함)를 영상에서 찾아 홀더 위치를 구한다 — 순수 계산.

ArUco 22 mm 는 640x480 에서 검출되지 않지만(2026-10-13), 흰 패드는 20 화소 안팎의 밝은 사각형이라 보인다.
전제(`sim2real/config/cup_holders.yaml`): 홀더는 똑바로 서 있고, 받침 바닥이 상판(z 0.205) 위, 클램프가 상판 끝에 닿아
yaw = 0(패드 법선 -x). 그러면 패드 중심의 높이를 알고, 영상 한 점의 광선과 그 높이 평면의 교점이 패드 중심(x, y)이다.
홀더 원점 = 패드 중심 - PAD_CENTER_IN_HOLDER.

한계: 패드를 마커 종이·볼트 반사와 혼동하면 틀린다(면적·종횡비·밝기로만 거른다). yaw 가 0 이 아니면 x 가 어긋난다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: cup_holders.yaml — STL 원점(받침 윗면, 컵 축 위) 기준 패드 중심. m.
PAD_CENTER_IN_HOLDER = np.array([-0.041, 0.0, -0.015])
#: 받침 바닥이 상판 위, 바닥은 원점 아래 30 mm → 원점 z.
HOLDER_ORIGIN_Z = 0.205 + 0.030
#: 홀더 x 는 클램프가 상판 끝에 닿아 모두 같다(cup_holders.yaml shared: [x], sim 0.380). 패드는 그 x - 0.041 의 선 위에 있다.
HOLDER_X = 0.380
PAD_SIZE_M = 0.030
MIN_DENSITY = 0.35
MAX_DENSITY = 0.85  # 상자가 통째로 흰 곳(벽·바닥)은 패드가 아니다
MAX_SURROUND_DENSITY = 0.45  # 패드는 검은 홀더·상판에 둘러싸여 있다(구슬 반사 여유를 둔 값, 실측 조정)
#: 패드는 그늘져 어둡다(V 130 안팎) — 절대 밝기가 아니라 낮은 채도 + 최소 밝기로 거른다.
MIN_VAL = 70
MAX_SAT = 70
SCAN_STEP_M = 0.005
PEAK_SEPARATION_M = 0.06


@dataclass(frozen=True)
class Pad:
    """검출된 패드: 중심 화소, 후보 y(base, m), 상자 안 흰 화소 밀도."""

    u: float
    v: float
    y_scan: float
    density: float


def _project(t_base_cam: np.ndarray, k: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    inv = np.linalg.inv(t_base_cam)
    cam = pts @ inv[:3, :3].T + inv[:3, 3]
    if np.any(cam[:, 2] <= 1e-3):
        return None
    return np.column_stack([k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2], k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]])


def _pad_corners(y: float) -> np.ndarray:
    """패드(수직면, 법선 -x)의 네 모서리 base 좌표."""
    x = HOLDER_X + PAD_CENTER_IN_HOLDER[0]
    z0 = HOLDER_ORIGIN_Z + PAD_CENTER_IN_HOLDER[2]
    h = PAD_SIZE_M / 2
    return np.array([[x, y + dy, z0 + dz] for dy in (-h, h) for dz in (-h, h)])


def detect_pads(rgb: np.ndarray, t_base_cam: np.ndarray, k: np.ndarray) -> list[Pad]:
    """홀더 패드가 놓일 선을 따라 패드 크기의 상자를 쓸며, 안쪽 흰 화소 밀도가 높은 극대점을 패드로 돌려준다(y 오름차순).

    상자 크기는 자세에서 패드 모서리를 투영해 정하므로 목 각도에 따라 자동으로 바뀐다.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    white = ((hsv[..., 2] >= MIN_VAL) & (hsv[..., 1] <= MAX_SAT)).astype(np.float32)
    height, width = white.shape
    ys = np.arange(-0.45, 0.45 + 1e-9, SCAN_STEP_M)
    scan = []  # (y, density, 상자 u0, v0, u1, v1)
    for y in ys:
        uv = _project(t_base_cam, k, _pad_corners(y))
        if uv is None:
            continue
        u0, v0 = np.floor(uv.min(0)).astype(int)
        u1, v1 = np.ceil(uv.max(0)).astype(int)
        if u0 < 0 or v0 < 0 or u1 >= width or v1 >= height or u1 - u0 < 6 or v1 - v0 < 4:
            continue
        inner = white[v0:v1 + 1, u0:u1 + 1]
        pad_w, pad_h = u1 - u0 + 1, v1 - v0 + 1
        eu0, ev0 = max(u0 - pad_w // 2, 0), max(v0 - pad_h // 2, 0)
        eu1, ev1 = min(u1 + pad_w // 2, width - 1), min(v1 + pad_h // 2, height - 1)
        outer = white[ev0:ev1 + 1, eu0:eu1 + 1]
        ring_px = outer.size - inner.size
        surround = (outer.sum() - inner.sum()) / ring_px if ring_px > 0 else 1.0
        density = float(inner.mean()) if surround <= MAX_SURROUND_DENSITY else 0.0
        scan.append((float(y), density, u0, v0, u1, v1))
    pads = []
    for i, (y, dens, u0, v0, u1, v1) in enumerate(scan):
        if not MIN_DENSITY <= dens <= MAX_DENSITY:
            continue
        near = [d for (yy, d, *_rest) in scan if abs(yy - y) <= PEAK_SEPARATION_M]
        if dens < max(near):
            continue
        ys_idx, xs_idx = np.nonzero(white[v0:v1 + 1, u0:u1 + 1])
        pads.append(Pad(u=float(u0 + xs_idx.mean()), v=float(v0 + ys_idx.mean()), y_scan=y, density=dens))
    unique = []
    for pad in pads:  # 같은 극대점이 이웃 y 에서 반복되는 것을 하나로
        if not unique or abs(pad.y_scan - unique[-1].y_scan) > PEAK_SEPARATION_M:
            unique.append(pad)
    return unique


def ray_plane_xy(t_base_cam: np.ndarray, k: np.ndarray, uv: tuple[float, float], z_plane: float) -> np.ndarray:
    """화소 (u, v) 의 광선과 base z=z_plane 평면의 교점 (x, y)."""
    d = np.array([(uv[0] - k[0, 2]) / k[0, 0], (uv[1] - k[1, 2]) / k[1, 1], 1.0])
    direction = t_base_cam[:3, :3] @ d
    origin = t_base_cam[:3, 3]
    return (origin + direction * ((z_plane - origin[2]) / direction[2]))[:2]


def holder_xy_from_pad(pad: Pad, t_base_cam: np.ndarray, k: np.ndarray) -> np.ndarray:
    """패드 중심 → 홀더 원점 (x, y). yaw 0 가정."""
    pad_z = HOLDER_ORIGIN_Z + PAD_CENTER_IN_HOLDER[2]
    return ray_plane_xy(t_base_cam, k, (pad.u, pad.v), pad_z) - PAD_CENTER_IN_HOLDER[:2]
