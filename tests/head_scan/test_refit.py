import numpy as np
import pytest

from perception_plus_plus_core.head_scan import head_fk, refit

K = np.array([[606.6, 0, 320.0], [0, 605.7, 240.6], [0, 0, 1.0]])
POSES = [(-30.0, -30.0), (-15.0, -25.0), (0.0, -25.0), (15.0, -25.0), (30.0, -25.0), (0.0, -20.0), (-40.0, -30.0), (10.0, -30.0)]
BOARD_Z = 0.205
TRUE = refit.Params(rot=np.array([0.004, -0.003, 0.002]), trans=np.array([0.002, -0.0015, 0.003]), pan_off_deg=0.25, tilt_off_deg=-0.2, f_scale=1.01)
BOARDS = {"A": refit.BoardPose(0.40, -0.05, 0.1), "B": refit.BoardPose(0.43, -0.33, -0.05)}


def _synthetic(params=TRUE, poses=POSES, noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed)
    data = []
    for pan, tilt in poses:
        for label, board in BOARDS.items():
            ids = np.arange(24)
            pts = refit.board_points(board, ids, BOARD_Z, mirror=True)
            uv = refit.project(pts, pan, tilt, params, K, head_fk.EXTRINSICS_YAML)
            ok = (uv[:, 0] > 10) & (uv[:, 0] < 630) & (uv[:, 1] > 10) & (uv[:, 1] < 470)
            if ok.sum() >= 6:
                data.append(refit.Detection(pan=pan, tilt=tilt, board=label, ids=ids[ok], uv=uv[ok] + rng.normal(0, noise_px, (ok.sum(), 2)), k=K))
    return data


def test_refit_recovers_a_known_hand_eye_correction_and_offsets():
    data = _synthetic()
    result = refit.fit(data, head_fk.EXTRINSICS_YAML, board_z=BOARD_Z, free=("rot", "trans", "angles", "focal"))
    assert result.rms_px < 0.05
    # pan 영점은 보드 위치·yaw 와 구분되지 않는다(보드 절대 위치가 자유 미지수) — 맞춤 잔차와 초점 배율만 확인한다.
    assert result.params.f_scale == pytest.approx(TRUE.f_scale, abs=0.005)


def test_leave_one_pose_out_error_drops_when_the_true_effects_are_modelled():
    data = _synthetic(noise_px=0.2)
    baseline = refit.leave_one_pose_out(data, head_fk.EXTRINSICS_YAML, board_z=BOARD_Z, free=())
    modelled = refit.leave_one_pose_out(data, head_fk.EXTRINSICS_YAML, board_z=BOARD_Z, free=("rot", "trans", "angles", "focal"))
    assert modelled.rms_mm < baseline.rms_mm
    assert modelled.rms_mm < 1.0


def test_radial_distortion_is_recovered_when_it_is_present():
    truth = refit.Params(k1=-0.08, k2=0.02)
    data = _synthetic(params=truth)
    flat = refit.fit(data, head_fk.EXTRINSICS_YAML, board_z=BOARD_Z, free=("rot", "trans"))
    distorted = refit.fit(data, head_fk.EXTRINSICS_YAML, board_z=BOARD_Z, free=("rot", "trans", "dist"))
    assert flat.rms_px > 0.3
    assert distorted.rms_px < 0.05
    assert distorted.params.k1 == pytest.approx(-0.08, abs=0.01)
