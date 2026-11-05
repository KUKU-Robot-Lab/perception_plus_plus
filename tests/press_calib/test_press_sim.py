import numpy as np
import pytest

from perception_plus_plus_core import press_calib as pc
from perception_plus_plus_core.press_calib import sim as S

TABLE_Z = 0.205
BOARD = S.board_pose(0.20, -0.30, TABLE_Z, 0.0)  # 기본 5 코너가 모두 닿는 자리


@pytest.fixture(scope="module")
def model():
    return S.HandModel()


@pytest.fixture(scope="module")
def plans(model):
    return S.plan_session(model, BOARD, TABLE_Z)


def test_hand_model_matches_robot_control_chain(model):
    fk = pc.FingertipFK()
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.uniform(-0.5, 0.5, 7)
        iq = float(rng.uniform(0, 1))
        np.testing.assert_allclose(model.link_pos(pc.TIP_LINK, S.hand_q(q, iq)), fk.tip(q, iq), atol=1e-9)


def test_ik_reaches_and_points_down(model):
    r = S.solve_press_ik(model, np.array([0.3, -0.2, TABLE_Z]), np.array([0, 0, -1.0]))
    assert r.ok and r.pos_err_m < 1e-4 and r.dir_err_deg < 1.0


def test_unreachable_target_flagged(model):
    assert not S.solve_press_ik(model, np.array([0.45, -0.2, 0.0]), np.array([0, 0, -1.0])).ok


def test_default_corners_all_pass(plans):
    assert [p.corner_id for p in plans] == pc.default_corner_ids()
    assert all(p.ok for p in plans), [p.problems for p in plans]
    assert all(p.min_clear_m > 0.01 for p in plans)


def test_closed_loop_recovers_board_and_injected_error(model, plans):
    cam = np.eye(4)
    cam[:3, 3] = [0.1, 0.0, 1.2]
    rep = S.closed_loop(model, plans, BOARD, cam, inject_pos_m=0.005, inject_rot_deg=1.0)
    assert rep.board_pos_err_m < 5e-5 and rep.board_rot_err_deg < 0.01
    assert abs(rep.detected_rot_deg - rep.injected_rot_deg) < 0.01
    assert abs(rep.detected_pos_m - rep.injected_pos_m) < 1e-4
    assert rep.corrected_pos_err_m < 1e-4 and rep.corrected_rot_err_deg < 0.01


def test_sensitivity_shrinks_with_repeats(model, plans):
    one = S.sensitivity(model, plans, BOARD, 0.002, 0.002, trials=60, repeats=1)
    three = S.sensitivity(model, plans, BOARD, 0.002, 0.002, trials=60, repeats=3)
    assert three.pos_p50_mm < one.pos_p50_mm
    zero = S.sensitivity(model, plans, BOARD, 0.0, 0.0, trials=5)
    assert zero.pos_p95_mm < 0.05


def test_substitute_replaces_failed_corner(model):
    # (0.22, -0.32) 에서는 코너 5 가 팔을 거의 편 자세라 하강 중 팔꿈치가 0.4 rad 움직여 탈락 → 다른 코너로 바뀐다
    T = S.board_pose(0.22, -0.32, TABLE_Z, 0.0)
    ids, plans = S.substitute_corners(model, T, TABLE_Z)
    assert 5 not in ids and len(ids) == 5
    assert all(p.ok for p in plans) and len(set(ids)) == len(ids)
