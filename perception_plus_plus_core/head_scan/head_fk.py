"""목 FK — 실기 인코더 각(deg)으로 `base ← camera_color_optical_frame` 을 계산한다.

    T_base_cam(pan, tilt) = T_base_neck(pan, tilt) ∘ T_neck_cam

관절 값은 `config/robot_camera_extrinsics/head_chain.yaml`(URDF 사본), hand-eye 는 같은 폴더의
`head_extrinsics.yaml` 에서 읽는다. sim2real 의 `head_fk_chain`·`head_camera_pose` 와 같은 결과를 내며,
어긋나면 테스트가 잡는다. ROS·URDF 없이 돈다(인지 PC 에서 git pull 만으로 쓰기 위해).
회전·쿼터니언 보조 함수는 sim2real 의 복사본이고, 동기화는 `test_matches_sim2real_base_cam_pose` 가 지킨다.
(sim2real `head_fk_chain` 은 `openarm_tesollo_sensor_rl.urdf` 를 읽는다. 목 체인 값은 두 URDF 가 같다.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "robot_camera_extrinsics"
CHAIN_YAML = CONFIG_DIR / "head_chain.yaml"
EXTRINSICS_YAML = CONFIG_DIR / "head_extrinsics.yaml"

#: 인코더 각 → URDF 관절 각. pan 만 부호가 반대다(URDF pan 축이 (0,0,-1)).
#: 근거는 sim2real `head_fk_chain.PAN_ENCODER_TO_URDF` 의 2026-09-01 hand-eye 판정.
PAN_ENCODER_TO_URDF = -1.0
TILT_ENCODER_TO_URDF = +1.0

REVOLUTE_KINDS = ("revolute", "continuous")


@dataclass(frozen=True)
class Joint:
    """URDF 관절 하나(부모 기준 origin xyz/rpy, 회전축)."""

    name: str
    kind: str
    xyz: tuple[float, ...]
    rpy: tuple[float, ...]
    axis: tuple[float, ...]


@lru_cache(maxsize=4)
def load_chain(path: Path | str = CHAIN_YAML) -> tuple[Joint, ...]:
    """목 체인 관절들(부모 → 자식 순). 프로세스 안에서 캐시되므로 yaml 을 바꾸면 재시작해야 한다."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        return tuple(
            Joint(name=j["name"], kind=j["type"], xyz=tuple(map(float, j["xyz"])),
                  rpy=tuple(map(float, j["rpy"])), axis=tuple(map(float, j.get("axis", (0, 0, 1)))))
            for j in raw["joints"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: joints 형식이 틀렸다") from exc


@lru_cache(maxsize=4)
def load_neck_to_camera(path: Path | str = EXTRINSICS_YAML) -> np.ndarray:
    """hand-eye `T_neck_cam`(4x4, 읽기 전용). 캐시되므로 yaml 을 바꾸면 재시작해야 한다."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        matrix = np.array(raw["neck_to_camera"]["matrix"], dtype=float)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: neck_to_camera.matrix 가 없다") from exc
    if matrix.shape != (4, 4):
        raise ValueError(f"{path}: neck_to_camera.matrix 는 4x4 여야 한다 — {matrix.shape}")
    matrix.setflags(write=False)
    return matrix


def urdf_from_encoder(pan_deg: float, tilt_deg: float) -> tuple[float, float]:
    """실기 인코더 각(deg) → URDF 관절 각(deg)."""
    return (PAN_ENCODER_TO_URDF * float(pan_deg), TILT_ENCODER_TO_URDF * float(tilt_deg))


def _rpy_matrix(rpy: tuple[float, ...]) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle_matrix(axis: tuple[float, ...], angle_rad: float) -> np.ndarray:
    a = np.array(axis, dtype=float)
    norm = np.linalg.norm(a)
    if norm < 1e-12:
        return np.eye(3)
    a = a / norm
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * k + (1 - math.cos(angle_rad)) * (k @ k)


def _homogeneous(rot: np.ndarray, trans) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = trans
    return out


def t_base_neck(pan_encoder_deg: float, tilt_encoder_deg: float) -> np.ndarray:
    """`base ← head_camera`(tilt 링크) 4x4. 입력은 실기 인코더 각(deg)."""
    pan_urdf, tilt_urdf = urdf_from_encoder(pan_encoder_deg, tilt_encoder_deg)
    angles = {"head_j_pan": math.radians(pan_urdf), "head_j_tilt": math.radians(tilt_urdf)}
    transform = np.eye(4)
    for joint in load_chain():
        transform = transform @ _homogeneous(_rpy_matrix(joint.rpy), joint.xyz)
        if joint.kind in REVOLUTE_KINDS:
            if joint.name not in angles:
                raise RuntimeError(f"각도를 모르는 관절: {joint.name}")
            transform = transform @ _homogeneous(
                _axis_angle_matrix(joint.axis, angles[joint.name]), (0, 0, 0))
    return transform


def t_base_cam(pan_encoder_deg: float, tilt_encoder_deg: float) -> np.ndarray:
    """`base ← camera_color_optical_frame` 4x4."""
    return t_base_neck(pan_encoder_deg, tilt_encoder_deg) @ load_neck_to_camera()


def quat_wxyz_from_matrix(rot: np.ndarray) -> np.ndarray:
    """회전행렬 → (w,x,y,z). 완전 직교가 아니면 가장 가까운 회전으로 투영한다."""
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=float))
    m = u @ vt
    trace = np.trace(m)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    quat = np.array(q, dtype=float)
    return quat / np.linalg.norm(quat)


def base_cam_pose(pan_encoder_deg: float, tilt_encoder_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(base 기준 카메라 위치 3, 쿼터니언 wxyz 4)."""
    transform = t_base_cam(pan_encoder_deg, tilt_encoder_deg)
    return transform[:3, 3].copy(), quat_wxyz_from_matrix(transform[:3, :3])
