"""목 FK(head_scan.head_fk) 검증 — 기준 자세, URDF 사본 일치, sim2real 원본 일치."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

from perception_plus_plus_core.head_scan import head_fk

REPO = Path(__file__).resolve().parents[2]
RL_WS = REPO.parent
EXTRINSICS = REPO / "config" / "robot_camera_extrinsics" / "head_extrinsics.yaml"
URDF = RL_WS / "urdf" / "generated" / "rl" / "openarm_rh56f1_bi_rl.urdf"
SIM2REAL_SCRIPTS = RL_WS / "sim2real" / "scripts"

POSES = [(0.0, -20.0), (0.0, -30.0), (15.0, -20.0), (-15.0, -20.0), (30.0, -45.0)]


def _quat_close(a, b, tol=1e-6):
    a, b = np.asarray(a), np.asarray(b)
    return min(np.abs(a - b).max(), np.abs(a + b).max()) < tol


def test_home_pose_matches_reference_in_extrinsics_yaml():
    ref = yaml.safe_load(EXTRINSICS.read_text(encoding="utf-8"))["reference_pose"]
    pos, quat = head_fk.base_cam_pose(ref["pan_deg"], ref["tilt_deg"])
    assert np.allclose(pos, ref["position"], atol=1e-6)
    assert _quat_close(quat, ref["orientation_wxyz"])


def test_pan_encoder_sign_is_flipped_for_urdf():
    assert head_fk.urdf_from_encoder(10.0, -20.0) == (-10.0, -20.0)


def test_rotation_part_is_orthonormal():
    for pan, tilt in POSES:
        T = head_fk.t_base_cam(pan, tilt)
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.skipif(not URDF.exists(), reason="원본 URDF 없음(인지 PC)")
def test_chain_yaml_matches_source_urdf():
    root = ET.parse(URDF).getroot()
    by_name = {j.get("name"): j for j in root.findall("joint")}
    for joint in head_fk.load_chain():
        src = by_name[joint.name]
        origin = src.find("origin")
        assert np.allclose(joint.xyz, [float(v) for v in origin.get("xyz").split()], atol=1e-9)
        assert np.allclose(joint.rpy, [float(v) for v in origin.get("rpy").split()], atol=1e-9)
        axis = src.find("axis")
        if axis is not None:
            assert np.allclose(joint.axis, [float(v) for v in axis.get("xyz").split()], atol=1e-9)
        assert joint.kind == src.get("type")


@pytest.mark.skipif(not SIM2REAL_SCRIPTS.exists(), reason="sim2real 없음(인지 PC)")
@pytest.mark.parametrize("pan,tilt", POSES)
def test_matches_sim2real_base_cam_pose(pan, tilt):
    sys.path.insert(0, str(SIM2REAL_SCRIPTS))
    try:
        from head_camera_pose import base_cam_pose as ref_fn
    finally:
        sys.path.remove(str(SIM2REAL_SCRIPTS))
    ref_pos, ref_quat = ref_fn(pan, tilt)
    pos, quat = head_fk.base_cam_pose(pan, tilt)
    assert np.allclose(pos, ref_pos, atol=1e-6)
    assert _quat_close(quat, ref_quat)


def test_positive_encoder_pan_turns_camera_forward_axis_toward_plus_y():
    """독립 부호 검증(URDF·sim2real 없이도 돈다). URDF pan 축 (0,0,-1) + 2026-09-01 hand-eye 판정으로
    인코더 + 는 위에서 볼 때 반시계 → 전방(+x)이 +y 쪽으로 간다."""
    forward_home = head_fk.t_base_cam(0.0, -20.0)[:3, :3] @ np.array([0.0, 0.0, 1.0])
    forward_pan = head_fk.t_base_cam(30.0, -20.0)[:3, :3] @ np.array([0.0, 0.0, 1.0])
    # 인코더 tilt 0 이 바로 아래이므로 tilt -20 의 수평 성분은 sin(20°)≈0.34, pan 30° 의 y 몫은 그 절반 ≈ 0.17.
    assert forward_home[1] == pytest.approx(0.0, abs=1e-6)
    assert forward_pan[1] == pytest.approx(0.17, abs=0.02)


def test_neck_to_camera_is_read_only():
    with pytest.raises(ValueError):
        head_fk.load_neck_to_camera()[0, 0] = 9.0


def test_t_base_cam_accepts_alternative_extrinsics_file(tmp_path):
    import numpy as np

    from perception_plus_plus_core.head_scan import head_fk

    shifted = head_fk.load_neck_to_camera().copy()
    shifted[0, 3] += 0.01
    rows = "\n".join("    - [" + ", ".join(map(str, r)) + "]" for r in shifted)
    path = tmp_path / "alt.yaml"
    path.write_text(f"neck_to_camera:\n  matrix:\n{rows}\n", encoding="utf-8")
    base = head_fk.t_base_cam(10.0, -20.0)
    alt = head_fk.t_base_cam(10.0, -20.0, path)
    assert np.linalg.norm(alt[:3, 3] - base[:3, 3]) == __import__("pytest").approx(0.01, abs=1e-9)
