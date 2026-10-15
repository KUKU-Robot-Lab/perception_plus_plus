"""테이블 위에 선 물체의 x, y 를 CAD 실루엣과 색 마스크의 일치로 구한다 — 순수 계산(로봇·GPU 없음).

전제: 물체는 상판 위에 똑바로 서 있다 → z(= 상판 + 원점 높이), roll, pitch 는 아는 값이다. 대칭축이 z 인 원통형 물체는
yaw 도 무의미하므로 미지수는 x, y 둘뿐이다. 카메라 자세는 목 FK(또는 볼트 PnP)로 주어진다고 본다.
FPP 포즈는 초기값으로만 쓰고, 최적화가 그 값을 마스크 일치로 바로잡는다(깊이 편향·z 오차를 이 단계에서 제거).

한계: 마스크가 가려지거나 같은 색 배경이 섞이면 틀린다. IoU 가 낮으면 값을 믿지 말 것(`FitResult.iou`).
렌더는 삼각형 채우기라 조명·그림자·깊이 가림을 보지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

#: OpenCV HSV(H 0~179) 판정 기본값 — 채도·명도 하한과 hue 허용폭. 물체별 hue 중심은 호출자가 준다.
DEFAULT_HUE_TOL = 10
DEFAULT_MIN_SAT = 80
DEFAULT_MIN_VAL = 50


@dataclass(frozen=True)
class FitResult:
    """x, y 는 base 좌표(m). iou 는 최종 실루엣과 마스크의 일치도(0~1), init_iou 는 시작값에서의 일치도."""

    x: float
    y: float
    iou: float
    init_iou: float


def hue_mask(rgb: np.ndarray, hue: float, tol: float = DEFAULT_HUE_TOL, min_sat: int = DEFAULT_MIN_SAT,
             min_val: int = DEFAULT_MIN_VAL) -> np.ndarray:
    """RGB 영상에서 hue(OpenCV 0~179, 원형) 가 중심±tol 이고 채도·명도가 충분한 화소의 불리언 마스크."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    diff = np.abs(hsv[..., 0].astype(float) - hue)
    diff = np.minimum(diff, 180.0 - diff)
    return (diff <= tol) & (hsv[..., 1] >= min_sat) & (hsv[..., 2] >= min_val)


def largest_blob_near(mask: np.ndarray, uv: tuple[float, float], min_area: int = 200) -> np.ndarray:
    """마스크의 연결 성분 중 (u, v) 에서 가장 가까운(중심 기준) 것만 남긴다. 없으면 빈 마스크."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    best, best_d = 0, np.inf
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        d = float(np.hypot(centroids[i][0] - uv[0], centroids[i][1] - uv[1]))
        if d < best_d:
            best, best_d = i, d
    if not best:
        return np.zeros_like(mask, dtype=bool)
    blob = (labels == best).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blob, contours, -1, 1, thickness=cv2.FILLED)  # 열린 컵 안쪽처럼 색이 다른 구멍을 메운다
    return blob.astype(bool)


def project_points(points_base: np.ndarray, t_base_cam: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """base 좌표 점들 → (N x 2 화소, N 깊이)."""
    inv = np.linalg.inv(t_base_cam)
    cam = np.asarray(points_base, dtype=float) @ inv[:3, :3].T + inv[:3, 3]
    z = cam[:, 2]
    safe = np.where(z > 1e-6, z, 1e-6)
    return np.column_stack([k[0, 0] * cam[:, 0] / safe + k[0, 2], k[1, 1] * cam[:, 1] / safe + k[1, 2]]), z


def render_silhouette(vertices: np.ndarray, faces: np.ndarray, t_base_cam: np.ndarray, k: np.ndarray,
                      shape: tuple[int, int]) -> np.ndarray:
    """base 좌표 메시(꼭짓점, 삼각형)의 실루엣을 shape=(높이, 너비) 불리언 영상으로 그린다."""
    uv, z = project_points(vertices, t_base_cam, k)
    canvas = np.zeros(shape, dtype=np.uint8)
    # 삼각형마다 np.all 을 부르던 파이썬 루프가 렌더 시간의 대부분이었다 — 가시 삼각형을 한 번에 거르고 그리기만 반복한다.
    visible = (z[faces] > 1e-3).all(axis=1)
    for tri_uv in np.round(uv[faces[visible]]).astype(np.int32):
        cv2.fillConvexPoly(canvas, tri_uv, 1)
    return canvas.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def fit_xy(vertices_body: np.ndarray, faces: np.ndarray, z_origin: float, t_base_cam: np.ndarray, k: np.ndarray,
           mask: np.ndarray, xy_init: tuple[float, float], search_m: float = 0.03,
           occluder: tuple[np.ndarray, np.ndarray, float] | None = None) -> FitResult:
    """원점이 (x, y, z_origin), 자세 항등(Z-up)인 메시의 실루엣이 mask 와 가장 겹치는 (x, y) 를 찾는다.

    xy_init 에서 ±search_m 안에서만 찾는다(마스크가 다른 물체를 잡았을 때 멀리 달아나지 않게).
    occluder=(꼭짓점, 삼각형, 원점 z) 는 물체와 같은 (x, y) 에 서서 앞을 가리는 것(홀더 링 등)이다 — 그 실루엣 안의 화소는
    맞춤에서 뺀다(가려진 부분을 마스크에 없다고 벌하지 않게).
    """
    def cost(p):
        verts = vertices_body + np.array([p[0], p[1], z_origin])
        sil, target = render_silhouette(verts, faces, t_base_cam, k, mask.shape), mask
        if occluder is not None:
            occ = render_silhouette(occluder[0] + np.array([p[0], p[1], occluder[2]]), occluder[1], t_base_cam, k, mask.shape)
            sil, target = sil & ~occ, mask & ~occ
        return -iou(sil, target)

    init = np.asarray(xy_init, dtype=float)
    init_iou = -cost(init)

    def grid_search(center, half_width, step):
        offsets = np.arange(-half_width, half_width + 1e-9, step)
        return min((center + np.array([dx, dy]) for dx in offsets for dy in offsets), key=cost)

    best = grid_search(init, search_m, 0.01)  # 거친 격자 → 미세 격자 → Nelder-Mead
    best = grid_search(best, 0.01, 0.002)
    res = minimize(cost, best, method="Nelder-Mead", options={"xatol": 2e-4, "fatol": 1e-5})
    if res.fun < cost(best):
        best = res.x
    if np.linalg.norm(best - init) > search_m * 1.5:  # 마스크가 다른 것을 잡았을 가능성 — 시작값을 돌려준다
        best = init
    return FitResult(x=float(best[0]), y=float(best[1]), iou=float(-cost(best)), init_iou=float(init_iou))
