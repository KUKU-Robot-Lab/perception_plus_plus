"""(A) 정적 외부 파라미터 vs (B) 목 FK — 자세 간 흩어짐 비교(head_scan.compare) 검증."""
from __future__ import annotations

import numpy as np
import pytest

from perception_plus_plus_core.head_scan import compare, head_fk

STATIC_POSE = (0.0, -20.0)
TRUE_OBJ = np.eye(4)
TRUE_OBJ[:3, 3] = [0.30, -0.10, 0.25]


def _record(pan, tilt):
    """진짜 물체를 (pan, tilt) 자세의 카메라가 본 기록."""
    t_cam_obj = np.linalg.inv(head_fk.t_base_cam(pan, tilt)) @ TRUE_OBJ
    return compare.Record(name="cup", pan_deg=pan, tilt_deg=tilt, t_cam_obj=t_cam_obj)


def test_fk_is_consistent_across_head_poses_while_static_is_not():
    t_static = head_fk.t_base_cam(*STATIC_POSE)
    records = [_record(0, -20), _record(15, -20), _record(-15, -20), _record(0, -30)]
    result = compare.compare(records, t_static)["cup"]
    assert result.spread_fk_m == pytest.approx(0.0, abs=1e-9)
    assert result.spread_static_m > 0.02
    assert result.mean_fk == pytest.approx(TRUE_OBJ[:3, 3], abs=1e-9)


def test_single_pose_at_static_pose_gives_zero_difference():
    t_static = head_fk.t_base_cam(*STATIC_POSE)
    result = compare.compare([_record(0, -20)], t_static)["cup"]
    assert result.max_static_vs_fk_m == pytest.approx(0.0, abs=1e-9)


def test_fk_error_from_a_wrong_pan_offset_shows_up_as_spread():
    t_static = head_fk.t_base_cam(*STATIC_POSE)
    good = [_record(0, -20), _record(20, -20)]
    wrong = compare.Record(name="cup", pan_deg=-20 + 3.0, tilt_deg=-20,  # 기록한 pan 이 3° 어긋난 관측
                           t_cam_obj=np.linalg.inv(head_fk.t_base_cam(-20, -20)) @ TRUE_OBJ)
    result = compare.compare(good + [wrong], t_static)["cup"]
    assert result.spread_fk_m > 0.01


def test_empty_gives_empty():
    assert compare.compare([], np.eye(4)) == {}
