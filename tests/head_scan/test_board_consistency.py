import numpy as np
import pytest

from perception_plus_plus_core.head_scan import board_consistency as bc, head_fk

K = np.array([[606.6, 0, 320.0], [0, 605.7, 240.6], [0, 0, 1.0]])
BOARD_Z = 0.200
ORIGIN = np.array([0.345, -0.155])  # 보드 첫 코너(base xy)


def _corner_xy(ids):
    ids = np.asarray(ids)
    col, row = ids % (bc.SX - 1), ids // (bc.SX - 1)
    return ORIGIN + np.column_stack([(row + 1) * bc.SQUARE, (col + 1) * bc.SQUARE])  # 보드 x 축이 base y, y 축이 base x 에 가깝다는 가정은 필요 없다


def _project(t_base_cam, xy, z):
    pts = np.column_stack([xy, np.full(len(xy), z)])
    inv = np.linalg.inv(t_base_cam)
    cam = pts @ inv[:3, :3].T + inv[:3, 3]
    return np.column_stack([K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2], K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]])


def _obs(pan, tilt, ids, fk_pan=None):
    """진짜 자세(pan, tilt)에서 본 픽셀. fk_pan 이 다르면 분석이 쓰는 FK 각도만 틀어진다(인코더 오차 흉내)."""
    true_t = head_fk.t_base_cam(pan, tilt)
    uv = _project(true_t, _corner_xy(ids), BOARD_Z)
    return bc.PoseObs(t_base_cam=head_fk.t_base_cam(fk_pan if fk_pan is not None else pan, tilt), k=K, ids=np.asarray(ids), uv=uv)


IDS = np.arange(24)


def test_consistent_fk_gives_zero_spread_and_recovers_board_height():
    report = bc.consistency([_obs(p, -25.0, IDS) for p in (-20.0, 0.0, 15.0)])
    assert report.best_z == pytest.approx(BOARD_Z, abs=1e-3)
    assert report.rms_at_best_m < 1e-4
    assert report.n_common == 24


def test_encoder_error_in_one_pose_shows_up_as_spread_and_pose_offset():
    report = bc.consistency([_obs(-20.0, -25.0, IDS), _obs(0.0, -25.0, IDS, fk_pan=0.3), _obs(15.0, -25.0, IDS)])
    assert report.rms_at_best_m > 5e-4  # z 를 자유롭게 풀면 오차 일부를 흡수한다(0.3° 가 1 mm 로 보인다)
    offsets = np.linalg.norm(report.pose_offset_m, axis=1)
    assert offsets.argmax() == 1


def test_partial_visibility_uses_only_corners_seen_in_two_or_more_poses():
    a = _obs(-20.0, -25.0, np.arange(0, 16))
    b = _obs(0.0, -25.0, np.arange(8, 24))
    report = bc.consistency([a, b])
    assert report.n_common == 8


def test_grid_scale_is_one_for_a_correct_pose_and_flags_scale_error():
    obs = _obs(0.0, -25.0, IDS)
    scale, rms = bc.grid_scale(obs, BOARD_Z)
    assert scale == pytest.approx(1.0, abs=1e-3) and rms < 1e-4
    scale_wrong, _ = bc.grid_scale(obs, BOARD_Z + 0.03)  # 높이를 3 cm 틀리게 가정하면 배율이 어긋난다
    assert abs(scale_wrong - 1.0) > 0.01
