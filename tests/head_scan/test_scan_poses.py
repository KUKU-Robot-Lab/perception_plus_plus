"""스캔 자세 후보 계산(head_scan.scan_poses) 검증."""
from __future__ import annotations

import numpy as np
import pytest

from perception_plus_plus_core.head_scan import head_fk, scan_poses


def test_table_points_cover_whole_rectangle_at_table_height():
    pts = scan_poses.table_points(step_m=0.05)
    assert pts[:, 2] == pytest.approx(scan_poses.TABLE_TOP_Z)
    assert pts[:, 0].min() == pytest.approx(scan_poses.TABLE_X[0])
    assert pts[:, 0].max() == pytest.approx(scan_poses.TABLE_X[1])
    assert pts[:, 1].min() == pytest.approx(scan_poses.TABLE_Y[0])
    assert pts[:, 1].max() == pytest.approx(scan_poses.TABLE_Y[1])


def test_point_in_front_of_camera_is_visible_and_point_behind_is_not():
    cam = scan_poses.default_camera()
    pose = head_fk.t_base_cam(0.0, -20.0)
    forward = pose[:3, 3] + pose[:3, :3] @ np.array([0.0, 0.0, 0.6])
    behind = pose[:3, 3] - pose[:3, :3] @ np.array([0.0, 0.0, 0.6])
    mask = scan_poses.visible_mask(np.array([forward, behind]), pose, cam)
    assert mask.tolist() == [True, False]


def test_point_nearer_than_min_range_is_not_visible():
    cam = scan_poses.default_camera(min_range_m=0.3)
    pose = head_fk.t_base_cam(0.0, -20.0)
    near = pose[:3, 3] + pose[:3, :3] @ np.array([0.0, 0.0, 0.2])
    assert not scan_poses.visible_mask(near[None, :], pose, cam)[0]


def test_greedy_cover_reaches_target_and_never_repeats_a_pose():
    pts = scan_poses.table_points(step_m=0.05)
    result = scan_poses.greedy_scan_set(pts, pan_range=(-60, 60), tilt_range=(-60, 0),
                                        step_deg=10.0, target=0.9)
    assert result.covered_fraction >= 0.9
    assert len(set(result.poses)) == len(result.poses)
    assert len(result.poses) >= 1


def test_greedy_cover_reports_unreachable_target_instead_of_looping():
    pts = scan_poses.table_points(step_m=0.05)
    result = scan_poses.greedy_scan_set(pts, pan_range=(0, 0), tilt_range=(-20, -20),
                                        step_deg=10.0, target=1.0)
    assert result.covered_fraction < 1.0
    assert len(result.poses) == 1


def test_projection_pixel_directions_and_values():
    cam = scan_poses.default_camera(edge_margin_px=0.0, min_range_m=0.1)
    ident = np.eye(4)  # 카메라 광학 프레임 == base 프레임으로 둔다
    u, v, z = scan_poses.project(np.array([[0.1, 0.0, 0.6], [0.0, 0.1, 0.6], [0.0, 0.0, 0.6]]), ident, cam)
    assert u[0] == pytest.approx(cam.cx + cam.fx * 0.1 / 0.6)  # +x → 오른쪽(u 증가), fx 사용
    assert v[1] == pytest.approx(cam.cy + cam.fy * 0.1 / 0.6)  # +y → 아래(v 증가), fy 사용
    assert (u[2], v[2]) == pytest.approx((cam.cx, cam.cy))
    assert z.tolist() == [0.6, 0.6, 0.6]


def test_edge_margin_decides_visibility_at_the_border():
    cam = scan_poses.default_camera(edge_margin_px=40.0, min_range_m=0.1)
    ident = np.eye(4)
    inside = np.array([[(cam.width - 41.0 - cam.cx) / cam.fx * 0.6, 0.0, 0.6]])
    outside = np.array([[(cam.width - 39.0 - cam.cx) / cam.fx * 0.6, 0.0, 0.6]])
    assert scan_poses.visible_mask(inside, ident, cam)[0]
    assert not scan_poses.visible_mask(outside, ident, cam)[0]


def test_greedy_picks_positive_encoder_pan_for_a_point_on_the_left_side_of_the_table():
    # 인코더 pan + 는 카메라 전방을 +y 쪽으로 돌린다(head_fk 테스트의 pan 부호 단언과 같은 근거).
    point = np.array([[0.30, 0.40, scan_poses.TABLE_TOP_Z]])
    result = scan_poses.greedy_scan_set(point, pan_range=(-60, 60), tilt_range=(-60, 0), step_deg=10.0, target=1.0)
    assert result.covered_fraction == 1.0
    assert result.poses[0][0] > 0


def test_grid_never_exceeds_range_and_rejects_bad_step():
    assert scan_poses._grid((0, 13), 5.0) == [0, 5.0, 10.0]
    with pytest.raises(ValueError):
        scan_poses._grid((0, 10), 0.0)
