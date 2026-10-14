import importlib.util
from pathlib import Path

import numpy as np
import pytest

from perception_plus_plus_core.head_scan import head_fk

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/calibrate_profile.py"


@pytest.fixture(scope="module")
def cal():
    if not Path("/home/user/rl_ws/sim2real/scripts/calib/table_cad_extrinsics.py").exists():
        pytest.skip("sim2real 이 없는 환경")
    spec = importlib.util.spec_from_file_location("calibrate_profile_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corr_between_is_the_inverse_of_applying_the_correction(cal):
    base = head_fk.load_neck_to_camera()
    corr = cal.corr_matrix(np.array([0.01, -0.02, 0.015, 0.004, -0.003, 0.002]))
    target = base @ corr
    again = base @ cal.corr_matrix(cal.corr_between(base, target))
    assert np.allclose(again, target, atol=1e-9)


def test_group_bias_returns_the_mean_cad_minus_estimate_per_commanded_pose(cal):
    """추정이 CAD 보다 일정하게 (+2, -1) mm 치우쳐 있으면 그 자세의 보정은 (-2, +1) mm 여야 한다."""
    from perception_plus_plus_core.head_scan import refit

    hand_eye = head_fk.load_neck_to_camera()
    pan, tilt = 10.0, -25.0
    k = np.array([[606.6, 0, 320.0], [0, 605.7, 240.6], [0, 0, 1.0]])
    holes_xy = np.array([[0.2, -0.1], [0.25, 0.0], [0.3, 0.1], [0.22, 0.12], [0.35, -0.05], [0.28, -0.12]])
    t = head_fk.t_base_cam(pan, tilt)
    pts = np.column_stack([holes_xy, np.full(len(holes_xy), cal.Z)])
    inv = np.linalg.inv(t)
    cam = pts @ inv[:3, :3].T + inv[:3, 3]
    img = np.column_stack([k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2], k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]])
    obj = np.column_stack([holes_xy + np.array([0.002, -0.001]), np.full(len(holes_xy), cal.Z)])  # CAD 가 추정보다 (+2, -1) mm
    frame = {"commanded": (10.0, -25.0), "pan": pan, "tilt": tilt, "k": k, "img": img, "obj": obj}
    groups = cal.group_bias([frame, {**frame}], hand_eye)
    g = groups[(10.0, -25.0)]
    assert g["n_frames"] == 2 and g["n_bolts"] == 12
    assert g["bias"] == pytest.approx([0.002, -0.001], abs=1e-6)
    assert refit  # 임포트 확인


def test_frames_with_too_few_bolts_are_left_out_of_the_bias_table(cal):
    hand_eye = head_fk.load_neck_to_camera()
    frame = {"commanded": (0.0, -30.0), "pan": 0.0, "tilt": -30.0, "k": np.eye(3), "img": np.zeros((3, 2)), "obj": np.zeros((3, 3))}
    assert cal.group_bias([frame], hand_eye) == {}
