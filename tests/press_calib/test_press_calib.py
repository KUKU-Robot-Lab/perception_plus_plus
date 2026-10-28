import numpy as np
import pytest

from perception_plus_plus_core import press_calib as pc


def _pose(yaw=0.3, t=(0.4, -0.2, 0.21)):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


def test_corner_grid():
    P = pc.corner_points_board()
    assert P.shape == (24, 3)
    np.testing.assert_allclose(P[0], [0.03, 0.03, 0])
    np.testing.assert_allclose(P[23], [0.18, 0.12, 0])


def test_plan_targets_above_board():
    ts = pc.plan_press_targets(_pose(), approach_m=0.03)
    assert [t.corner_id for t in ts] == pc.default_corner_ids()
    for t in ts:
        np.testing.assert_allclose(t.approach_base - t.contact_base, [0, 0, 0.03], atol=1e-12)
    assert pc.check_targets(ts, table_z=0.21) == []
    assert pc.check_targets(ts, table_z=0.30)


def test_plan_rejects_bad_id():
    with pytest.raises(ValueError):
        pc.plan_press_targets(_pose(), [24])


def test_fit_recovers_pose_with_noise():
    T = _pose()
    ids = pc.default_corner_ids()
    rng = np.random.default_rng(0)
    tips = pc.plan_press_targets(T, ids)
    Q = np.array([t.contact_base for t in tips]) + rng.normal(0, 0.0005, (len(ids), 3))
    fit = pc.fit_board_pose(ids, Q)
    assert np.abs(fit.T_base_board[:3, 3] - T[:3, 3]).max() < 0.002
    assert fit.rms_m < 0.002


def test_fit_rejects_collinear():
    with pytest.raises(ValueError):
        pc.fit_board_pose([0, 1, 2], np.zeros((3, 3)))


def test_compare_with_camera_detects_offset():
    T_bb = _pose()
    T_bc = _pose(yaw=1.0, t=(0.1, 0.0, 0.8))
    T_cb = np.linalg.inv(T_bc) @ T_bb
    T_bc_bad = T_bc.copy()
    T_bc_bad[:3, 3] += [0.01, 0, 0]
    chk = pc.compare_with_camera(T_bc_bad, T_cb, T_bb)
    np.testing.assert_allclose(chk.dpos_m, [0.01, 0, 0], atol=1e-9)
    np.testing.assert_allclose(chk.T_base_cam_corrected, T_bc, atol=1e-9)


def test_fingertip_fk_runs():
    fk = pc.FingertipFK()
    p = fk.tip(np.zeros(7))
    assert p.shape == (3,) and np.isfinite(p).all()
    assert fk.out_of_limits(np.zeros(7)) == []
    assert fk.out_of_limits(np.full(7, 9.0))


def test_fk_matches_independent_urdf_walk():
    fk = pc.FingertipFK()
    T = pc._fixed_chain_to_root(pc.DEFAULT_URDF.read_text(), pc.TIP_LINK)
    np.testing.assert_allclose(fk.tip(np.zeros(7)), T[:3, 3], atol=1e-9)
