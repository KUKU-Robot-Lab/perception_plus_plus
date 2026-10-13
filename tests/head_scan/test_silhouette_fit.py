import numpy as np
import pytest
import trimesh

from perception_plus_plus_core.head_scan import head_fk, silhouette_fit as sf

K = np.array([[606.6, 0, 320.0], [0, 605.7, 240.6], [0, 0, 1.0]])
SHAPE = (480, 640)
Z0 = 0.205 + 0.065


def _cylinder():
    mesh = trimesh.creation.cylinder(radius=0.0354, height=0.13, sections=48)  # 원점 = 중심, 축 z
    return mesh.vertices, mesh.faces


def test_fit_xy_recovers_known_position_from_offset_start():
    verts, faces = _cylinder()
    t_bc = head_fk.t_base_cam(10.0, -20.0)
    truth = np.array([0.22, 0.05])
    mask = sf.render_silhouette(verts + np.r_[truth, Z0], faces, t_bc, K, SHAPE)
    fit = sf.fit_xy(verts, faces, Z0, t_bc, K, mask, tuple(truth + [0.012, -0.009]))
    assert np.hypot(fit.x - truth[0], fit.y - truth[1]) < 1.5e-3
    assert fit.iou > 0.95 and fit.iou > fit.init_iou


def test_fit_xy_returns_start_when_mask_is_far_away():
    verts, faces = _cylinder()
    t_bc = head_fk.t_base_cam(0.0, -20.0)
    mask = sf.render_silhouette(verts + np.r_[0.30, -0.20, Z0], faces, t_bc, K, SHAPE)
    fit = sf.fit_xy(verts, faces, Z0, t_bc, K, mask, (0.22, 0.10), search_m=0.02)
    assert (fit.x, fit.y) == pytest.approx((0.22, 0.10))


def test_hue_mask_wraps_around_and_fills_holes():
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[10:30, 10:30] = (255, 0, 0)  # 빨강: hue 0 (원형 경계)
    rgb[18:22, 18:22] = (0, 0, 0)    # 가운데 구멍
    m = sf.hue_mask(rgb, hue=178.0)
    assert m[12, 12] and not m[20, 20]
    filled = sf.largest_blob_near(m, (20, 20), min_area=10)
    assert filled[20, 20]
