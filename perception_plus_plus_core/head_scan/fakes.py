"""로봇 없이 `locate` 흐름을 돌리는 가짜 목·카메라 — 테스트와 `locate_object.py --fake` 용.

가짜 카메라는 가짜 목이 마지막으로 받은 자세에서 CAD 실루엣을 렌더해 영상을 만든다(물체가 시야 밖이면 빈 영상).
"""

from __future__ import annotations

import numpy as np

from . import head_fk, locate, silhouette_fit, survey

DEFAULT_K = np.array([[606.6, 0, 320.0], [0, 605.7, 240.6], [0, 0, 1.0]])
IMAGE_SHAPE = (480, 640)


class FakeHead:
    """명령한 각도를 그대로 실측으로 돌려주고 이동·복귀 호출을 기록한다. fail_on_move=n 이면 n 번째 이동에서 예외."""

    def __init__(self, fail_on_move: int | None = None):
        self.moves: list[tuple[float, float]] = []
        self.homed = 0
        self.fail_on_move = fail_on_move

    def move(self, pan_deg: float, tilt_deg: float) -> survey.HeadReading:
        if self.fail_on_move is not None and len(self.moves) == self.fail_on_move:
            raise RuntimeError("목 이동 실패")
        self.moves.append((pan_deg, tilt_deg))
        return survey.HeadReading(pan_deg=pan_deg, tilt_deg=tilt_deg, ok=True)

    def go_home(self) -> survey.HeadReading:
        self.homed += 1
        return survey.HeadReading(pan_deg=locate.HOME_POSE[0], tilt_deg=locate.HOME_POSE[1], ok=True)


class FakeCamera:
    """truth_xy(base, m)에 선 물체를 렌더한다. color 는 RGB, table_z 는 상판 높이."""

    def __init__(self, head: FakeHead, truth_xy: tuple[float, float], spec: locate.ObjectSpec, color: tuple[int, int, int],
                 table_z: float = 0.205, k: np.ndarray = DEFAULT_K):
        self.head, self.truth, self.spec, self.color, self.table_z, self.k = head, truth_xy, spec, color, table_z, k

    def capture(self, n_frames: int = 1) -> list[locate.Frame]:
        pan, tilt = self.head.moves[-1]
        verts = self.spec.vertices + np.r_[self.truth, self.table_z + self.spec.origin_above_bottom]
        sil = silhouette_fit.render_silhouette(verts, self.spec.faces, head_fk.t_base_cam(pan, tilt), self.k, IMAGE_SHAPE)
        rgb = np.full((*IMAGE_SHAPE, 3), 20, np.uint8)
        rgb[sil] = self.color
        return [locate.Frame(rgb=rgb, k=self.k, stamp_s=float(i)) for i in range(n_frames)]
