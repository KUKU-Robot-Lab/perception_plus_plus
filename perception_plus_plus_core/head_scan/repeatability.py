"""오차 분해 실험 — 같은 물체를 (a) 같은 자세에서 여러 프레임, (b) 다른 자세로 갔다가 같은 자세로 돌아와 여러 번 찍어
촬영 잡음과 목 재현성을 나눠 잰다. 물체와 카메라는 그 동안 움직이지 않아야 한다. 순수 로직(목·카메라는 포트).

frame_noise_m: 같은 방문 안에서 프레임마다 맞춘 (x, y) 의 흩어짐(평균 표준편차 아님, 방문 안 최대 거리의 평균).
visit_scatter_m: 방문별 평균 (x, y) 사이의 최대 거리 — 목이 같은 명령으로 돌아와도 생기는 차이(인코더·기구 재현성 포함).
angle_scatter_deg: 방문마다 실측한 pan 각도의 최대-최소.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import head_fk, locate, silhouette_fit, standing_pose


@dataclass(frozen=True)
class Visit:
    pan_deg: float
    tilt_deg: float
    fits: tuple[tuple[float, float], ...]  # 프레임마다 맞춘 (x, y); 못 맞춘 프레임은 빠진다


@dataclass(frozen=True)
class Report:
    visits: tuple[Visit, ...]
    frame_noise_m: float
    visit_scatter_m: float
    angle_scatter_deg: float
    home_ok: bool


def _max_dist(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2).max())


def run(head: locate.HeadPort, camera: locate.CameraPort, spec: locate.ObjectSpec, pose: tuple[float, float], detour: tuple[float, float],
        n_visits: int = 5, n_frames: int = 5, table_z: float = standing_pose.TABLE_TOP_Z,
        extrinsics_yaml=head_fk.EXTRINSICS_YAML) -> Report:
    """pose 로 가서 찍고, detour 로 갔다가 pose 로 돌아와 다시 찍기를 n_visits 번 반복한다. 끝나면 기준 자세로 복귀."""
    from . import survey

    survey.validate_poses([pose, detour, locate.HOME_POSE])
    z_origin = table_z + spec.origin_above_bottom
    visits: list[Visit] = []
    try:
        for i in range(n_visits):
            if i:
                head.move(*detour)
            reading = head.move(*pose)
            t_bc = head_fk.t_base_cam(reading.pan_deg, reading.tilt_deg, extrinsics_yaml)
            fits = []
            for frame in camera.capture(n_frames):
                init = locate.best_blob_init(spec, frame.rgb, frame.k, t_bc, z_origin)
                if init is None:
                    continue
                uv, _ = silhouette_fit.project_points(np.array([[init[0], init[1], z_origin]]), t_bc, frame.k)
                mask = silhouette_fit.largest_blob_near(silhouette_fit.hue_mask(frame.rgb, spec.hue), tuple(uv[0]))
                if mask.sum() < standing_pose.MIN_MASK_PX:
                    continue
                fit = silhouette_fit.fit_xy(spec.vertices, spec.faces, z_origin, t_bc, frame.k, mask, init)
                if fit.iou >= standing_pose.DEFAULT_MIN_IOU:
                    fits.append((fit.x, fit.y))
            visits.append(Visit(reading.pan_deg, reading.tilt_deg, tuple(fits)))
    finally:
        home_ok = bool(head.go_home().ok)
    noise = [_max_dist(np.array(v.fits)) for v in visits if v.fits]
    means = np.array([np.mean(v.fits, axis=0) for v in visits if v.fits])
    pans = [v.pan_deg for v in visits]
    return Report(visits=tuple(visits), frame_noise_m=float(np.mean(noise)) if noise else float("nan"),
                  visit_scatter_m=_max_dist(means) if len(means) else float("nan"),
                  angle_scatter_deg=float(max(pans) - min(pans)) if pans else float("nan"), home_ok=home_ok)
