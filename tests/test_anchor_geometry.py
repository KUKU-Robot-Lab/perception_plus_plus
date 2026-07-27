# tests/test_anchor_geometry.py
import numpy as np
from perception_plus_plus_core.tracking.anchor_geometry import (
    reproject, inside_box, anchor_valid, bbox_depth_mask,
)

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])


def _pose(x, y, z):
    p = np.eye(4)
    p[:3, 3] = (x, y, z)
    return p


def test_reproject_center():
    uv = reproject(_pose(0.0, 0.0, 1.0), K)
    assert uv is not None
    assert abs(uv[0] - 320.0) < 1e-6 and abs(uv[1] - 240.0) < 1e-6


def test_reproject_behind_camera_is_none():
    assert reproject(_pose(0.0, 0.0, -0.5), K) is None


def test_inside_box_margin():
    box = (300.0, 220.0, 340.0, 260.0)
    assert inside_box((320.0, 240.0), box, 0.0)
    assert not inside_box((360.0, 240.0), box, 0.0)      # 밖
    assert inside_box((360.0, 240.0), box, 0.5)          # margin 확장으로 안


def test_anchor_valid_z_gate():
    assert not anchor_valid(_pose(0, 0, 0.05), K, None, 0.15, 1.6, 0.35)   # 너무 가까움
    assert not anchor_valid(_pose(0, 0, 2.0), K, None, 0.15, 1.6, 0.35)    # 너무 멀음
    assert anchor_valid(_pose(0, 0, 0.6), K, None, 0.15, 1.6, 0.35)        # z OK, det 없음 → 유효


def test_anchor_valid_off_cup():
    # pose가 화면 중앙(320,240)에 투영되는데 det bbox는 우측에 있음 → 벗어남
    det = (500.0, 220.0, 560.0, 300.0)
    assert not anchor_valid(_pose(0, 0, 0.6), K, det, 0.15, 1.6, 0.35)
    det_center = (300.0, 220.0, 340.0, 260.0)
    assert anchor_valid(_pose(0, 0, 0.6), K, det_center, 0.15, 1.6, 0.35)


def test_bbox_depth_mask_selects_near_band():
    depth = np.full((480, 640), 2.5, np.float32)   # 배경 원거리
    depth[230:250, 310:330] = 0.6                  # bbox 안 근거리 컵
    mask = bbox_depth_mask(depth, (300, 220, 340, 260))
    assert mask.dtype == bool and mask.shape == (480, 640)
    assert mask[240, 320]            # 근거리 픽셀 포함
    assert not mask[0, 0]            # bbox 밖 제외
