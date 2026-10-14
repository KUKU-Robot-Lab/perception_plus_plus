"""ChArUco 보드로 목 FK·hand-eye 의 자세 간 일관성을 잰다 — 순수 계산(검출은 `detect_corners`).

보드는 테이블에 고정돼 있다. 목 자세마다 코너 픽셀을 카메라 광선으로 바꿔 z=보드 높이 평면과 만나면 base xy 가 나온다.
FK·hand-eye 가 맞으면 자세가 달라도 같은 코너는 같은 xy 에 떨어진다. 자세 사이 흩어짐이 곧 FK·hand-eye 오차다.
보드를 어디에 놓았는지(절대 위치)는 필요 없다. 보드 높이는 모르면 흩어짐을 최소로 하는 값을 찾는다(best_z).
grid_scale 은 한 자세 안에서 코너 격자 간격이 규격(30 mm)과 맞는지 본다(내부 파라미터·거리 오차 점검).
보드 규격은 sim2real `scripts/charuco_calib.py`: 7x5, DICT_6X6_250, square 0.030, marker 0.022.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SX, SY, SQUARE, MARKER = 7, 5, 0.030, 0.022
DEFAULT_Z_GRID = tuple(np.arange(0.190, 0.2151, 0.0005))


@dataclass(frozen=True)
class PoseObs:
    """한 자세의 관측: base←camera 4x4, 내부 행렬, 검출된 코너 id 와 픽셀(N x 2)."""

    t_base_cam: np.ndarray
    k: np.ndarray
    ids: np.ndarray
    uv: np.ndarray


@dataclass(frozen=True)
class ConsistencyReport:
    best_z: float
    rms_at_best_m: float
    rms_by_z: dict[float, float]
    pose_offset_m: np.ndarray  # 자세별 (x, y): 그 자세의 코너 평균이 전체 합의에서 벗어난 양
    n_common: int


def detect_corners(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """RGB 영상에서 ChArUco 코너 (ids, uv). 4개 미만이면 None. OpenCV 4.5 API(`CharucoBoard_create`)."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard_create(SX, SY, SQUARE, MARKER, dictionary)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) < 2:
        return None
    count, charuco, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if charuco is None or count < 4:
        return None
    return charuco_ids.ravel().astype(int), charuco.reshape(-1, 2).astype(float)


def _plane_xy(obs: PoseObs, z: float) -> np.ndarray:
    ray_cam = np.column_stack([(obs.uv[:, 0] - obs.k[0, 2]) / obs.k[0, 0], (obs.uv[:, 1] - obs.k[1, 2]) / obs.k[1, 1], np.ones(len(obs.uv))])
    ray = ray_cam @ obs.t_base_cam[:3, :3].T
    s = (z - obs.t_base_cam[2, 3]) / ray[:, 2]
    return (obs.t_base_cam[:3, 3] + s[:, None] * ray)[:, :2]


def consistency(observations: list[PoseObs], z_grid=DEFAULT_Z_GRID) -> ConsistencyReport:
    """2개 이상의 자세에서 보인 코너만 써서 자세 간 흩어짐(RMS)을 z 후보마다 구하고 최소를 고른다."""
    if len(observations) < 2:
        raise ValueError("자세가 2개 이상 필요하다")
    seen: dict[int, int] = {}
    for obs in observations:
        for i in obs.ids:
            seen[int(i)] = seen.get(int(i), 0) + 1
    common = sorted(i for i, n in seen.items() if n >= 2)
    if not common:
        raise ValueError("두 자세 이상에서 보인 코너가 없다")
    rms_by_z = {}
    best = None
    for z in z_grid:
        rms, offsets = _spread(observations, common, float(z))
        rms_by_z[round(float(z), 4)] = rms
        if best is None or rms < best[0]:
            best = (rms, float(z), offsets)
    return ConsistencyReport(best_z=best[1], rms_at_best_m=best[0], rms_by_z=rms_by_z, pose_offset_m=best[2], n_common=len(common))


def _spread(observations: list[PoseObs], common: list[int], z: float) -> tuple[float, np.ndarray]:
    per_pose = []
    for obs in observations:
        xy = _plane_xy(obs, z)
        per_pose.append({int(i): xy[j] for j, i in enumerate(obs.ids)})
    consensus = {i: np.mean([p[i] for p in per_pose if i in p], axis=0) for i in common}
    sq, offsets = [], []
    for p in per_pose:
        diffs = np.array([p[i] - consensus[i] for i in common if i in p])
        sq.extend(np.sum(diffs ** 2, axis=1))
        offsets.append(diffs.mean(axis=0) if len(diffs) else np.zeros(2))
    return float(np.sqrt(np.mean(sq))), np.array(offsets)


def grid_scale(obs: PoseObs, z: float) -> tuple[float, float]:
    """한 자세의 코너를 z 평면에 얹은 뒤 규격 격자에 닮음변환(회전·이동·배율)으로 맞춘다 → (배율, 잔차 RMS m). 배율 1.000 이 정상."""
    xy = _plane_xy(obs, z)
    ids = obs.ids.astype(int)
    col, row = ids % (SX - 1), ids // (SX - 1)
    nominal = np.column_stack([(row + 1) * SQUARE, (col + 1) * SQUARE])
    mu_a, mu_b = xy.mean(axis=0), nominal.mean(axis=0)
    a, b = xy - mu_a, nominal - mu_b
    u, sig, vt = np.linalg.svd(b.T @ a / len(a))
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ d @ vt
    scale = float(np.trace(np.diag(sig) @ d) / np.mean(np.sum(b ** 2, axis=1)))
    fitted = (rot @ b.T).T * scale + mu_a
    return 1.0 / scale if scale else float("nan"), float(np.sqrt(np.mean(np.sum((fitted - xy) ** 2, axis=1))))
