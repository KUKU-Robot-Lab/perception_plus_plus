"""ChArUco 보드 영상으로 목 FK 모델을 다시 맞춘다(번들 조정 형태) — 순수 계산.

미지수: hand-eye 보정(회전 3·이동 3, 카메라 좌표계 기준 우곱), pan·tilt 영점 오프셋, 초점거리 배율, 보드마다 (x, y, yaw).
보드는 규격 격자(30 mm)라 보드 안 코너 간격이 척도 제약을 준다. 보드 높이는 아는 값으로 고정한다(테이블 + 인쇄물 두께).
잔차는 코너의 재투영 오차(픽셀)다. 풀 항목은 `free` 로 고른다: rot, trans, angles(pan·tilt 오프셋), focal, dist(k1·k2).
렌즈 왜곡은 방사 k1·k2(dist)만 둔다. 과적합은 `leave_one_pose_out` 으로 본다: 자세 하나를 빼고 맞춘 뒤 그 자세의 코너를 base 평면 위치로 바꿔 모델과 비교한다(mm).

한계: pan 영점 오프셋은 보드 (x, y, yaw) 와 구분되지 않는다(보드 절대 위치가 자유 미지수라서). 자세 간 일관성·예측에는 영향이 없지만
절대 base 좌표를 맞추려면 위치를 아는 기준(볼트 구멍 CAD 등)이 더 필요하다. 보드가 테이블 위 같은 높이에 있다는 가정, 렌즈 왜곡 0 가정(카메라가 0 으로 보고). 보드 절대 위치는 모른다(자유 미지수).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import board_consistency as bc, head_fk

FREE_ALL = ("rot", "trans", "angles", "focal", "dist")
BOARD_Z_FREE = "boardz"  # 보드마다 높이 오프셋(m)을 푼다


@dataclass(frozen=True)
class Params:
    rot: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 회전벡터(rad), 카메라 좌표계 기준 보정
    trans: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m
    pan_off_deg: float = 0.0
    tilt_off_deg: float = 0.0
    f_scale: float = 1.0
    k1: float = 0.0  # 방사 왜곡(정규화 좌표 기준)
    k2: float = 0.0


@dataclass(frozen=True)
class BoardPose:
    x: float
    y: float
    yaw: float
    dz: float = 0.0  # 기준 보드 높이에서의 오프셋(m)


@dataclass(frozen=True)
class Detection:
    """한 자세에서 본 한 보드: 인코더 실측 각도(deg), 보드 이름, 코너 id·픽셀, 내부 행렬."""

    pan: float
    tilt: float
    board: str
    ids: np.ndarray
    uv: np.ndarray
    k: np.ndarray


@dataclass(frozen=True)
class FitResult:
    params: Params
    boards: dict[str, BoardPose]
    rms_px: float
    mirror: bool


@dataclass(frozen=True)
class LopoResult:
    rms_mm: float
    per_pose_mm: dict[tuple[float, float], float]


def _rodrigues(vec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vec))
    if theta < 1e-12:
        return np.eye(3)
    a = vec / theta
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def board_points(board: BoardPose, ids: np.ndarray, z: float, mirror: bool) -> np.ndarray:
    """보드 코너(id)의 base 좌표(N x 3). 보드 좌표 (u=열, v=행)에서 mirror 이면 v 를 뒤집는다(위에서 본 평면과 거울상일 때)."""
    ids = np.asarray(ids).astype(int)
    u, v = ((ids % (bc.SX - 1)) + 1) * bc.SQUARE, ((ids // (bc.SX - 1)) + 1) * bc.SQUARE
    v = -v if mirror else v
    c, s = np.cos(board.yaw), np.sin(board.yaw)
    return np.column_stack([board.x + c * u - s * v, board.y + s * u + c * v, np.full(len(ids), z + board.dz)])


def camera_transform(pan: float, tilt: float, params: Params, extrinsics_yaml: Path | str) -> np.ndarray:
    corr = np.eye(4)
    corr[:3, :3], corr[:3, 3] = _rodrigues(params.rot), params.trans
    return head_fk.t_base_neck(pan + params.pan_off_deg, tilt + params.tilt_off_deg) @ head_fk.load_neck_to_camera(Path(extrinsics_yaml)) @ corr


def _scaled_k(k: np.ndarray, scale: float) -> np.ndarray:
    out = k.copy()
    out[0, 0] *= scale
    out[1, 1] *= scale
    return out


def project(points_base: np.ndarray, pan: float, tilt: float, params: Params, k: np.ndarray, extrinsics_yaml: Path | str) -> np.ndarray:
    inv = np.linalg.inv(camera_transform(pan, tilt, params, extrinsics_yaml))
    cam = points_base @ inv[:3, :3].T + inv[:3, 3]
    kk = _scaled_k(k, params.f_scale)
    xn, yn = cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2]
    r2 = xn ** 2 + yn ** 2
    factor = 1.0 + params.k1 * r2 + params.k2 * r2 ** 2
    return np.column_stack([kk[0, 0] * xn * factor + kk[0, 2], kk[1, 1] * yn * factor + kk[1, 2]])


def _undistort(xd: np.ndarray, yd: np.ndarray, k1: float, k2: float) -> tuple[np.ndarray, np.ndarray]:
    """왜곡된 정규화 좌표 → 왜곡 없는 좌표(고정점 반복)."""
    xu, yu = xd.copy(), yd.copy()
    for _ in range(8):
        r2 = xu ** 2 + yu ** 2
        factor = 1.0 + k1 * r2 + k2 * r2 ** 2
        xu, yu = xd / factor, yd / factor
    return xu, yu


def _ground_xy(det: Detection, params: Params, z: float, extrinsics_yaml: Path | str) -> np.ndarray:
    t = camera_transform(det.pan, det.tilt, params, extrinsics_yaml)
    kk = _scaled_k(det.k, params.f_scale)
    xu, yu = _undistort((det.uv[:, 0] - kk[0, 2]) / kk[0, 0], (det.uv[:, 1] - kk[1, 2]) / kk[1, 1], params.k1, params.k2)
    ray = np.column_stack([xu, yu, np.ones(len(det.uv))]) @ t[:3, :3].T
    return (t[:3, 3] + ((z - t[2, 3]) / ray[:, 2])[:, None] * ray)[:, :2]


def _init_board(dets: list[Detection], params: Params, z: float, mirror: bool, extrinsics_yaml: Path | str) -> BoardPose:
    """베이스라인 모델로 얻은 평면 위치에 규격 격자를 강체로 맞춰 보드 (x, y, yaw) 시작값을 만든다."""
    xy = np.vstack([_ground_xy(d, params, z, extrinsics_yaml) for d in dets])
    ids = np.concatenate([d.ids for d in dets]).astype(int)
    u, v = ((ids % (bc.SX - 1)) + 1) * bc.SQUARE, ((ids // (bc.SX - 1)) + 1) * bc.SQUARE
    nominal = np.column_stack([u, -v if mirror else v])
    mu_a, mu_b = xy.mean(axis=0), nominal.mean(axis=0)
    u_, _, vt = np.linalg.svd((xy - mu_a).T @ (nominal - mu_b))
    rot = u_ @ np.diag([1.0, np.sign(np.linalg.det(u_ @ vt))]) @ vt
    yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    trans = mu_a - rot @ mu_b
    return BoardPose(float(trans[0]), float(trans[1]), yaw)


def _pack(params: Params, boards: dict[str, BoardPose], names: list[str], free: tuple[str, ...]) -> np.ndarray:
    parts = []
    if "rot" in free:
        parts.append(params.rot)
    if "trans" in free:
        parts.append(params.trans)
    if "angles" in free:
        parts.append([params.pan_off_deg, params.tilt_off_deg])
    if "focal" in free:
        parts.append([params.f_scale - 1.0])
    if "dist" in free:
        parts.append([params.k1, params.k2])
    for n in names:
        parts.append([boards[n].x, boards[n].y, boards[n].yaw] + ([boards[n].dz] if BOARD_Z_FREE in free else []))
    return np.concatenate([np.asarray(p, dtype=float) for p in parts])


def _unpack(vec: np.ndarray, names: list[str], free: tuple[str, ...]) -> tuple[Params, dict[str, BoardPose]]:
    i = 0
    rot, trans, pan, tilt, f, k1, k2 = np.zeros(3), np.zeros(3), 0.0, 0.0, 1.0, 0.0, 0.0
    if "rot" in free:
        rot, i = vec[i:i + 3], i + 3
    if "trans" in free:
        trans, i = vec[i:i + 3], i + 3
    if "angles" in free:
        pan, tilt, i = float(vec[i]), float(vec[i + 1]), i + 2
    if "focal" in free:
        f, i = 1.0 + float(vec[i]), i + 1
    if "dist" in free:
        k1, k2, i = float(vec[i]), float(vec[i + 1]), i + 2
    width = 4 if BOARD_Z_FREE in free else 3
    boards = {n: BoardPose(float(vec[i + width * j]), float(vec[i + width * j + 1]), float(vec[i + width * j + 2]),
                           float(vec[i + width * j + 3]) if width == 4 else 0.0) for j, n in enumerate(names)}
    return Params(rot=rot, trans=trans, pan_off_deg=pan, tilt_off_deg=tilt, f_scale=f, k1=k1, k2=k2), boards


def fit(data: list[Detection], extrinsics_yaml: Path | str, board_z: float, free: tuple[str, ...] = FREE_ALL,
        mirror: bool | None = None) -> FitResult:
    """재투영 오차를 최소화한다. mirror=None 이면 두 경우를 모두 풀어 잔차가 작은 쪽을 쓴다."""
    if mirror is None:
        results = [fit(data, extrinsics_yaml, board_z, free, m) for m in (False, True)]
        return min(results, key=lambda r: r.rms_px)
    names = sorted({d.board for d in data})
    base = Params()
    boards0 = {n: _init_board([d for d in data if d.board == n], base, board_z, mirror, extrinsics_yaml) for n in names}
    x0 = _pack(base, boards0, names, free)

    def residual(vec):
        params, boards = _unpack(vec, names, free)
        return np.concatenate([(project(board_points(boards[d.board], d.ids, board_z, mirror), d.pan, d.tilt, params, d.k, extrinsics_yaml) - d.uv).ravel()
                               for d in data])

    sol = least_squares(residual, x0, method="lm" if len(x0) <= len(residual(x0)) else "trf")
    params, boards = _unpack(sol.x, names, free)
    return FitResult(params=params, boards=boards, rms_px=float(np.sqrt(np.mean(sol.fun ** 2))), mirror=mirror)


def leave_one_pose_out(data: list[Detection], extrinsics_yaml: Path | str, board_z: float, free: tuple[str, ...]) -> LopoResult:
    """자세 하나씩 빼고 맞춘 뒤, 뺀 자세의 코너를 평면 위치로 바꿔 보드 모델 위치와 비교한다(mm RMS)."""
    poses = sorted({(d.pan, d.tilt) for d in data})
    mirror = fit(data, extrinsics_yaml, board_z, free).mirror
    per_pose: dict[tuple[float, float], float] = {}
    for pose in poses:
        train = [d for d in data if (d.pan, d.tilt) != pose]
        test = [d for d in data if (d.pan, d.tilt) == pose]
        trained = {d.board for d in train}
        test = [d for d in test if d.board in trained]
        if not test:
            continue
        result = fit(train, extrinsics_yaml, board_z, free, mirror)
        errs = []
        for d in test:
            model_xy = board_points(result.boards[d.board], d.ids, board_z, mirror)[:, :2]
            errs.append(np.sum((_ground_xy(d, result.params, board_z + result.boards[d.board].dz, extrinsics_yaml) - model_xy) ** 2, axis=1))
        per_pose[pose] = float(np.sqrt(np.mean(np.concatenate(errs))) * 1000)
    values = np.array(list(per_pose.values()))
    return LopoResult(rms_mm=float(np.sqrt(np.mean(values ** 2))) if len(values) else float("nan"), per_pose_mm=per_pose)
