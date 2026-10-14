"""물체 찾기 요청 하나를 끝까지 처리한다 — 자세 계획, 목 이동, 촬영, 좌표 계산, 기준 자세 복귀. 순수 로직(로봇·ROS 없음).

목과 카메라는 포트(`HeadPort`, `CameraPort`)로 주입한다. 실기용 구현은 `adapters.py`, 테스트는 가짜를 쓴다.
흐름: 요청 → 후보 자세 → [이동·정지 → 촬영 → 색 마스크로 시작점 → 촬영 순간 목 각도 FK] 반복 →
`standing_pose.estimate_standing`(z 고정 실루엣 맞춤) → 결과. 충분히 일관된 관측이 모이면 남은 자세는 생략한다.

원칙: 못 찾으면 값을 만들지 않는다(found=False, x/y=None). 어떤 실패에서도 기준 자세로 복귀를 시도하고 결과에 복귀 성공 여부를 적는다.
FPP 는 쓰지 않는다 — 색 마스크 덩어리를 광선-평면 교점으로 바꿔 시작점으로 삼고, 정밀 값은 CAD 실루엣 맞춤이 낸다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from . import head_fk, scan_poses, silhouette_fit, standing_pose, survey

HOME_POSE = (0.0, -20.0)
StageCallback = Callable[[str, float, float], None]
DEFAULT_TILT_DEG = -20.0
DEFAULT_PAN_RANGE = (-90.0, 90.0)
DEFAULT_PAN_STEP = 10.0
#: 테이블 먼 가장자리를 화면에 넣으려면 더 위를 봐야 한다(tilt 더 음수). 이 tilt 에서 컵 검출이 흔들린 적이 있어 첫 후보는 아니다.
EXTRA_TILTS_DEG = (-30.0,)
#: 이 화소 수 미만의 색 덩어리는 물체로 보지 않는다.
MIN_BLOB_PX = 200
#: 서로 일관된 관측이 이만큼 모이고 흩어짐이 허용 이하면 더 찍지 않는다.
DEFAULT_MIN_USED = 2
DEFAULT_SPREAD_TOL_M = 0.003
DEFAULT_MAX_POSES = 5


@dataclass(frozen=True)
class ObjectSpec:
    """찾을 물체: CAD 메시(꼭짓점·삼각형, 원점 기준), 원점이 바닥에서 떠 있는 높이, OpenCV hue."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    origin_above_bottom: float
    hue: float


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray
    k: np.ndarray
    stamp_s: float


class HeadPort(Protocol):
    def move(self, pan_deg: float, tilt_deg: float) -> survey.HeadReading:
        """목을 보내고 정지한 뒤의 실측 각도를 돌려준다. 정지를 확인 못 하면 예외."""
        ...

    def go_home(self) -> survey.HeadReading:
        """기준 자세(pan 0, tilt -20)로 복귀."""
        ...


class CameraPort(Protocol):
    def capture(self, n_frames: int = 1) -> list[Frame]:
        """지금 자세에서 n 장 촬영."""
        ...


@dataclass(frozen=True)
class LocateRequest:
    """object_name 은 호출자 식별용. side 는 'right'/'left'/None(전체), poses 를 주면 계획 대신 그 자세만 쓴다."""

    object_name: str
    side: str | None = None
    poses: list[tuple[float, float]] | None = None
    prior_xy: tuple[float, float] | None = None
    tilt_deg: float = DEFAULT_TILT_DEG
    max_poses: int = DEFAULT_MAX_POSES
    min_used: int = DEFAULT_MIN_USED
    spread_tol_m: float = DEFAULT_SPREAD_TOL_M
    n_frames: int = 1


@dataclass(frozen=True)
class Shot:
    """촬영 한 번: 그때의 목 실측 각도(인코더 deg), 영상 시각, 이 자세에서 색 덩어리를 찾은 프레임 수."""

    pan_deg: float
    tilt_deg: float
    stamp_s: float
    n_blobs: int
    init_xy: tuple[float, float] | None = None  # 첫 색 덩어리의 광선-평면 교점(진단용)


@dataclass(frozen=True)
class LocateResult:
    found: bool
    x: float | None = None
    y: float | None = None
    z: float | None = None
    n_used: int = 0
    n_rejected: int = 0
    spread_m: float = float("nan")
    mean_iou: float = 0.0
    shots: tuple[Shot, ...] = field(default_factory=tuple)
    reason: str = ""
    home_ok: bool = True


def blob_init_xy(rgb: np.ndarray, k: np.ndarray, hue: float, t_base_cam: np.ndarray, z_origin: float) -> tuple[float, float] | None:
    """색 마스크의 가장 큰 덩어리 중심을 z=z_origin 평면과의 광선 교점으로 바꾼다. 없거나 평면이 뒤에 있으면 None."""
    mask = silhouette_fit.hue_mask(rgb, hue).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_PX]
    if not areas:
        return None
    u, v = centroids[max(areas)[1]]
    ray_cam = np.array([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1.0])
    ray = t_base_cam[:3, :3] @ ray_cam
    if ray[2] >= -1e-6:  # 평면(카메라보다 낮다)을 향하지 않는다
        return None
    s = (z_origin - t_base_cam[2, 3]) / ray[2]
    hit = t_base_cam[:3, 3] + s * ray
    return float(hit[0]), float(hit[1])


def plan_poses(side: str | None, tilt_deg: float = DEFAULT_TILT_DEG, pan_range: tuple[float, float] = DEFAULT_PAN_RANGE,
               pan_step: float = DEFAULT_PAN_STEP, extra_tilts: tuple[float, ...] = EXTRA_TILTS_DEG) -> list[tuple[float, float]]:
    """찾을 쪽 테이블을 덮는 자세를 앞에 두고(탐욕 덮기), 이어서 그 쪽이 많이 보이는 나머지 후보를 중복 시점용으로 붙인다.

    side: 'right'(y<0)·'left'(y>0)·None. 후보는 pan_range 안 pan_step 간격과 tilt_deg·extra_tilts 의 조합이다.
    tilt 한계(중력 쪽 -10°)는 `survey.validate_poses` 가 실행 전에 막는다.
    """
    if side not in (None, "right", "left"):
        raise ValueError(f"side 는 right/left/None: {side!r}")
    points = scan_poses.table_points()
    if side == "right":
        points = points[points[:, 1] < 0]
    elif side == "left":
        points = points[points[:, 1] > 0]
    cam = scan_poses.default_camera()
    tilts = [tilt_deg, *extra_tilts]
    candidates = [(pan, tilt) for tilt in tilts for pan in scan_poses._grid(pan_range, pan_step)]
    seen = {c: int(scan_poses.visible_mask(points, head_fk.t_base_cam(*c), cam).sum()) for c in candidates}
    cover = scan_poses.greedy_scan_set(points, pan_range, (min(tilts), max(tilts)), step_deg=pan_step, target=0.99, cam=cam)
    chosen = [tuple(map(float, c)) for c in cover.poses if c in seen]
    rest = sorted((c for c in candidates if c not in chosen and seen[c]), key=lambda c: (-seen[c], tilts.index(c[1]), abs(c[0])))
    return chosen + rest


def locate_object(request: LocateRequest, spec: ObjectSpec, head: HeadPort, camera: CameraPort, table_z: float = standing_pose.TABLE_TOP_Z,
                  extrinsics_yaml: Path | str = head_fk.EXTRINSICS_YAML, on_stage: StageCallback | None = None,
                  should_stop: Callable[[], bool] | None = None) -> LocateResult:
    """요청을 처리한다. 자세가 한계를 넘으면 움직이기 전에 ValueError. 그 밖의 실패는 결과의 reason 에 담는다.

    on_stage(단계, pan, tilt) 는 planning·moving·shooting·fitting·returning 순으로 불린다. should_stop() 이 참이면 다음 자세로
    가지 않고 복귀한다(취소). 둘 다 없어도 된다.
    """
    def stage(name: str, pan: float = 0.0, tilt: float = 0.0) -> None:
        if on_stage:
            on_stage(name, pan, tilt)

    stage("planning")
    poses = list(request.poses) if request.poses else plan_poses(request.side, request.tilt_deg)[:request.max_poses]
    survey.validate_poses(poses + [HOME_POSE])
    z_origin = table_z + spec.origin_above_bottom
    observations: list[standing_pose.StandingObs] = []
    shots: list[Shot] = []
    estimate = None
    reason = ""
    try:
        for pan, tilt in poses:
            if should_stop and should_stop():
                reason, estimate = "취소됨", None  # 취소는 부분 결과를 내지 않는다
                break
            stage("moving", pan, tilt)
            reading = head.move(pan, tilt)
            t_bc = head_fk.t_base_cam(reading.pan_deg, reading.tilt_deg, extrinsics_yaml)
            blobs, stamp, first_init = 0, 0.0, None
            stage("shooting", reading.pan_deg, reading.tilt_deg)
            for frame in camera.capture(request.n_frames):
                stamp = frame.stamp_s
                init = request.prior_xy or blob_init_xy(frame.rgb, frame.k, spec.hue, t_bc, z_origin)
                if init is None:
                    continue
                blobs += 1
                first_init = first_init or init
                observations.append(standing_pose.StandingObs(frame.rgb, frame.k, reading.pan_deg, reading.tilt_deg, init))
            shots.append(Shot(reading.pan_deg, reading.tilt_deg, stamp, blobs, first_init))
            if not blobs:
                continue
            stage("fitting", reading.pan_deg, reading.tilt_deg)
            estimate = standing_pose.estimate_standing(observations, spec.vertices, spec.faces, spec.origin_above_bottom, spec.hue,
                                                       extrinsics_yaml, base_z=table_z)
            if estimate and estimate.n_used >= request.min_used and estimate.spread_m <= request.spread_tol_m:
                break
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 복귀를 막지 않고 이유를 결과에 남긴다
        reason = str(exc)
        estimate = None
    home_ok = True
    stage("returning", *HOME_POSE)
    try:
        home_ok = bool(head.go_home().ok)
    except Exception as exc:  # noqa: BLE001
        home_ok, reason = False, f"{reason}; 복귀 실패: {exc}".lstrip("; ")
    if estimate is None:
        return LocateResult(found=False, shots=tuple(shots), reason=reason or "어느 자세에서도 물체를 못 찾았다", home_ok=home_ok)
    return LocateResult(found=True, x=estimate.x, y=estimate.y, z=estimate.z, n_used=estimate.n_used, n_rejected=estimate.n_rejected,
                        spread_m=estimate.spread_m, mean_iou=estimate.mean_iou, shots=tuple(shots), reason=reason, home_ok=home_ok)
