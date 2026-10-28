"""손끝으로 테이블 위 ChArUco 보드를 눌러 카메라 extrinsics 를 맞추는 모듈 (순수 numpy, 로봇 구동 없음).

흐름
    1. plan_press_targets   보드 자세(대략) → 누를 코너 목록(접근점 · 접촉점, base 프레임)
    2. (실기, 사용자 승인 뒤) 각 코너를 손끝으로 누르고 그 순간 팔 · 손 관절각을 기록
    3. FingertipFK          관절각 → 손끝 위치(base)
    4. fit_board_pose       (코너 id, 손끝 위치) 쌍 → T_base_board (Kabsch) + 잔차
    5. compare_with_camera  카메라가 본 T_cam_board 와 비교 → extrinsics 오차 / 보정 후보

상수 출처(두 저장소를 묶지 않으려고 복사만 함):
    보드   sim2real/scripts/charuco_calib.py  SX, SY, SQUARE, MARKER = 7, 5, 0.030, 0.022 · DICT_6X6_250 (origin 5846f88)
    링크   urdf/generated/rl/openarm_bi_rh56f1_rl.urdf  r_hl_index_tip · r_hj_index_2 mimic 1.1169 × r_hj_index_1
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ── 보드 (charuco_calib.py 와 같은 인쇄물) ─────────────────────────────
SX, SY, SQUARE, MARKER = 7, 5, 0.030, 0.022
N_CORNERS = (SX - 1) * (SY - 1)  # 24

# ── 로봇 (RH56F1 오른손) ───────────────────────────────────────────────
DEFAULT_URDF = Path("/home/user/rl_ws/urdf/generated/rl/openarm_bi_rh56f1_rl.urdf")
ROBOT_CONTROL_SRC = "/home/user/rl_ws/robot_control/src"
BASE_LINK = "body_root"
ARM_JOINTS = tuple(f"r_aj_{i}" for i in range(1, 8))
TIP_LINK = "r_hl_index_tip"
FINGER_JOINTS = ("r_hj_index_1", "r_hj_index_2")
INDEX_MIMIC = 1.1169  # r_hj_index_2 = 1.1169 × r_hj_index_1


def corner_points_board() -> np.ndarray:
    """ChArUco 내부 코너 (N_CORNERS, 3), 보드 프레임. id = row*(SX-1)+col (OpenCV 순서)."""
    ids = np.arange(N_CORNERS)
    col, row = ids % (SX - 1), ids // (SX - 1)
    return np.stack([(col + 1) * SQUARE, (row + 1) * SQUARE, np.zeros(N_CORNERS)], axis=1)


def _apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


@dataclass(frozen=True)
class PressTarget:
    corner_id: int
    contact_base: np.ndarray   # 보드 면 위 코너 (m, base)
    approach_base: np.ndarray  # 보드 법선 방향으로 approach_m 위


def default_corner_ids() -> list[int]:
    """네 모서리 + 가운데 — 회전 · 위치를 고르게 묶는 최소 세트."""
    nx = SX - 1
    return [0, nx - 1, N_CORNERS - nx, N_CORNERS - 1, (SY - 1) // 2 * nx + nx // 2]


def plan_press_targets(T_base_board: np.ndarray, corner_ids: Iterable[int] | None = None,
                       approach_m: float = 0.03) -> list[PressTarget]:
    """보드 자세 → 누를 목표. 법선은 base z 와 같은 쪽(위)으로 맞춘다."""
    T = np.asarray(T_base_board, float)
    ids = list(default_corner_ids() if corner_ids is None else corner_ids)
    bad = [i for i in ids if not 0 <= i < N_CORNERS]
    if bad:
        raise ValueError(f"코너 id 범위 밖: {bad} (0..{N_CORNERS - 1})")
    normal = T[:3, 2] if T[2, 2] >= 0 else -T[:3, 2]
    pts = _apply(T, corner_points_board()[ids])
    return [PressTarget(i, p, p + approach_m * normal) for i, p in zip(ids, pts)]


def check_targets(targets: Sequence[PressTarget], table_z: float,
                  min_clear_m: float = 0.0) -> list[str]:
    """테이블 아래로 가는 목표를 찾는다. 빈 리스트면 통과."""
    out = []
    for t in targets:
        if t.contact_base[2] < table_z - 0.005:
            out.append(f"corner {t.corner_id}: 접촉점 z {t.contact_base[2]:.3f} < 테이블 {table_z:.3f}")
        if t.approach_base[2] < table_z + min_clear_m:
            out.append(f"corner {t.corner_id}: 접근점이 테이블 여유 {min_clear_m} m 안")
    return out


class FingertipFK:
    """팔 7관절 + 검지 구동각 → 손끝 위치(base=body_root). robot_control.kinematics 체인 사용."""

    def __init__(self, urdf_path: Path | str = DEFAULT_URDF, tip_link: str = TIP_LINK) -> None:
        if ROBOT_CONTROL_SRC not in sys.path:
            sys.path.insert(0, ROBOT_CONTROL_SRC)
        from robot_control.kinematics import chain_from_urdf

        urdf = Path(urdf_path).read_text()
        self._chain = chain_from_urdf(urdf, [*ARM_JOINTS, *FINGER_JOINTS], tip_link)
        self.limits = _joint_limits(urdf, [*ARM_JOINTS, FINGER_JOINTS[0]])

    def tip(self, arm_q: Sequence[float], index_q: float = 0.0) -> np.ndarray:
        q = np.asarray(arm_q, float).reshape(-1)
        if q.size != 7:
            raise ValueError(f"arm_q 는 7개여야 한다: {q.size}")
        full = np.concatenate([q, [index_q, INDEX_MIMIC * index_q]])
        return self._chain.pose(full)[:3, 3]  # chain_from_urdf 는 URDF 루트(body_root)부터 고정 관절을 이미 포함

    def out_of_limits(self, arm_q: Sequence[float], index_q: float = 0.0) -> list[str]:
        vals = [*np.asarray(arm_q, float).reshape(-1), index_q]
        return [f"{n}={v:.3f} ∉ [{lo:.3f}, {hi:.3f}]"
                for (n, (lo, hi)), v in zip(self.limits.items(), vals) if not lo <= v <= hi]


def _fixed_chain_to_root(urdf: str, link: str) -> np.ndarray:
    """관절 0 자세에서 link → body_root (테스트 교차검증용, 모든 관절을 각 0 으로 본다)."""
    import xml.etree.ElementTree as ET
    by_child = {j.find("child").get("link"): j for j in ET.fromstring(urdf).iter("joint")}  # type: ignore[union-attr]
    T = np.eye(4)
    while link != BASE_LINK:
        j = by_child[link]
        o = j.find("origin")
        J = np.eye(4)
        J[:3, :3] = _rpy(*(float(v) for v in o.get("rpy", "0 0 0").split()))  # type: ignore[union-attr]
        J[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]  # type: ignore[union-attr]
        T = J @ T
        link = j.find("parent").get("link")  # type: ignore[union-attr]
    return T


def _joint_limits(urdf: str, names: Sequence[str]) -> dict[str, tuple[float, float]]:
    import xml.etree.ElementTree as ET
    joints = {j.get("name"): j for j in ET.fromstring(urdf).iter("joint")}
    out = {}
    for n in names:
        lim = joints[n].find("limit")
        out[n] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


@dataclass(frozen=True)
class BoardFit:
    T_base_board: np.ndarray
    residuals_m: np.ndarray  # 코너별 거리

    @property
    def rms_m(self) -> float:
        return float(np.sqrt(np.mean(self.residuals_m ** 2)))


def fit_board_pose(corner_ids: Sequence[int], tips_base: np.ndarray,
                   tip_radius_m: float = 0.0) -> BoardFit:
    """(코너 id, 손끝 위치) ≥3 쌍 → T_base_board (Kabsch).

    tip_radius_m: 손끝 링크 원점이 패드 중심이면 접촉면까지 반지름만큼 보드 법선(+z) 위에 있다고 본다.
    """
    ids = list(corner_ids)
    Q = np.asarray(tips_base, float).reshape(-1, 3)
    if len(ids) != len(Q) or len(ids) < 3:
        raise ValueError("코너 id 와 손끝 위치는 같은 개수, 3쌍 이상")
    P = corner_points_board()[ids] + np.array([0, 0, tip_radius_m])
    if np.linalg.matrix_rank(P - P.mean(0), tol=1e-6) < 2:
        raise ValueError("코너가 한 줄 위에 있다 — 회전이 정해지지 않는다")
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, qc - R @ pc
    return BoardFit(T, np.linalg.norm(_apply(T, P) - Q, axis=1))


@dataclass(frozen=True)
class CameraCheck:
    dpos_m: np.ndarray        # 카메라 경유 보드 위치 − 손끝 보드 위치 (base)
    drot_deg: float
    T_base_cam_corrected: np.ndarray  # 손끝 기준으로 맞춘 extrinsics 후보


def compare_with_camera(T_base_cam: np.ndarray, T_cam_board: np.ndarray,
                        T_base_board_touch: np.ndarray) -> CameraCheck:
    """카메라 경로(T_base_cam·T_cam_board)와 손끝 경로를 비교한다. 보정 후보는 저장하지 않고 돌려만 준다."""
    T_cam_path = np.asarray(T_base_cam) @ np.asarray(T_cam_board)
    T_touch = np.asarray(T_base_board_touch)
    dR = T_cam_path[:3, :3] @ T_touch[:3, :3].T
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    corrected = T_touch @ np.linalg.inv(T_cam_board)
    return CameraCheck(T_cam_path[:3, 3] - T_touch[:3, 3], float(ang), corrected)
