"""여러 목 자세에서 본 물체 포즈 통합(head_scan.merge_poses) 검증."""
from __future__ import annotations

import numpy as np
import pytest

from perception_plus_plus_core.head_scan import head_fk, merge_poses
from perception_plus_plus_core.head_scan.merge_poses import Observation

TRUE_POS = np.array([0.30, -0.10, 0.25])


def _true_pose() -> np.ndarray:
    t = np.eye(4)
    c, s = np.cos(0.4), np.sin(0.4)
    t[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t[:3, 3] = TRUE_POS
    return t


def _observe(pan, tilt, noise=None, name="cup") -> Observation:
    """진짜 base 포즈를 (pan, tilt) 자세의 카메라가 본 것처럼 만든다."""
    t_cam_obj = np.linalg.inv(head_fk.t_base_cam(pan, tilt)) @ _true_pose()
    if noise is not None:
        t_cam_obj = t_cam_obj.copy()
        t_cam_obj[:3, 3] += noise
    return Observation(name=name, pan_deg=pan, tilt_deg=tilt, t_cam_obj=t_cam_obj)


def test_noise_free_observations_from_different_poses_agree():
    obs = [_observe(10, -20), _observe(-30, -25), _observe(40, -35)]
    merged = merge_poses.merge(obs)["cup"]
    assert merged.position == pytest.approx(TRUE_POS, abs=1e-9)
    assert merged.pos_spread_m == pytest.approx(0.0, abs=1e-9)
    assert merged.n_used == 3 and merged.n_rejected == 0


def test_wrong_pan_sign_would_show_up_as_large_spread():
    good = [_observe(10, -20), _observe(40, -35)]
    bad_t = np.linalg.inv(head_fk.t_base_cam(-40, -35)) @ _true_pose()  # 부호를 뒤집어 기록한 관측
    bad = Observation(name="cup", pan_deg=40, tilt_deg=-35, t_cam_obj=bad_t)
    merged = merge_poses.merge(good + [bad], reject_m=10.0)["cup"]
    assert merged.n_rejected >= 1 or merged.pos_spread_m > 0.05


def test_outlier_is_rejected_and_estimate_stays_close():
    rng = np.random.default_rng(0)
    obs = [_observe(p, t, noise=rng.normal(0, 0.002, 3)) for p, t in [(0, -20), (20, -25), (-20, -25), (35, -30)]]
    obs.append(_observe(10, -20, noise=np.array([0.15, 0.0, 0.0])))
    merged = merge_poses.merge(obs, reject_m=0.03)["cup"]
    assert merged.n_rejected == 1
    assert np.linalg.norm(merged.position - TRUE_POS) < 0.005


def test_table_z_constraint_overrides_z_only():
    obs = [_observe(10, -20, noise=np.array([0, 0, 0.01]))]
    merged = merge_poses.merge(obs, table_z={"cup": 0.30})["cup"]
    assert merged.position[2] == pytest.approx(0.30)
    assert merged.position[:2] == pytest.approx(TRUE_POS[:2] + 0, abs=0.02)


def test_objects_are_merged_separately_and_rotation_recovers_yaw():
    obs = [_observe(10, -20, name="a"), _observe(40, -35, name="a"), _observe(10, -20, name="b")]
    merged = merge_poses.merge(obs)
    assert set(merged) == {"a", "b"}
    assert merged["a"].n_used == 2 and merged["b"].n_used == 1
    w, x, y, z = merged["a"].quat_wxyz
    assert 2 * np.arctan2(z, w) == pytest.approx(0.4, abs=1e-6)


def test_empty_input_gives_empty_result():
    assert merge_poses.merge([]) == {}
