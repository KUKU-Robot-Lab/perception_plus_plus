"""FK 사전 자세(후보 hand-eye)로 CAD 구멍을 영상에 투영하고, 각 구멍 둘레에서만 중심을 찾는다 — 짝짓기를 사전 자세로 못 박는다.

기존 `table_cad_extrinsics.solve` 는 투영 전체를 평행이동하며 대비 합이 최대인 곳을 찾는다(coarse_shift). 구멍이 2x2 로 반복돼
tilt -30 영상에서는 이웃 구멍으로 잘못 짝지어 100 mm 넘게 틀렸다(2026-10-14). 사전 자세 오차는 수 mm(= 수 px)이므로 구멍마다
작은 창 안에서만 찾고(`MAX_SHIFT_PX`), 창 안에 대비가 없으면 그 구멍은 버린다.
"""
import sys

import numpy as np

sys.path[:0] = ["/home/user/rl_ws/sim2real/scripts", "/home/user/rl_ws/sim2real/scripts/calib"]
import table_cad_extrinsics as T  # noqa: E402

MAX_SHIFT_PX = 12.0
MIN_R_PX = 3.0
MARGIN_PX = 12


def detect(gray: np.ndarray, k: np.ndarray, t_base_cam: np.ndarray, holes=None, max_shift_px: float = MAX_SHIFT_PX):
    """(검출 픽셀 N x 2, CAD xyz N x 3). 사전 자세로 투영한 구멍 중 창 안에서 중심을 찾은 것만."""
    holes = holes or T.cad_holes()
    xyz = np.array([h.xyz for h in holes])
    uv, depth = T.project(t_base_cam, k, xyz)
    height, width = gray.shape
    imgs, objs = [], []
    for hole, (u, v), z in zip(holes, uv, depth):
        r_px = hole.radius * k[0, 0] / z if z > 0 else 0.0
        if z <= 0 or r_px < MIN_R_PX or not (MARGIN_PX < u < width - MARGIN_PX and MARGIN_PX < v < height - MARGIN_PX):
            continue
        center = T.refine_center(gray, float(u), float(v), float(r_px))
        if center is None or np.hypot(center[0] - u, center[1] - v) > max_shift_px:
            continue
        imgs.append(center)
        objs.append(hole.xyz)
    return np.array(imgs).reshape(-1, 2), np.array(objs).reshape(-1, 3)
