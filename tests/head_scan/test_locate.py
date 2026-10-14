import cv2
import numpy as np
import pytest
import trimesh

from perception_plus_plus_core.head_scan import fakes, head_fk, locate

K = np.array([[606.6, 0, 320.0], [0, 605.7, 240.6], [0, 0, 1.0]])
HEIGHT = 0.13
ORIGIN_ABOVE_BOTTOM = HEIGHT / 2
BLUE_RGB = (30, 60, 220)
TABLE_Z = 0.205


def _hue():
    return float(cv2.cvtColor(np.array([[BLUE_RGB]], np.uint8), cv2.COLOR_RGB2HSV)[0, 0, 0])


def _spec():
    m = trimesh.creation.cylinder(radius=0.0354, height=HEIGHT, sections=48)
    return locate.ObjectSpec(name="blue_cup", vertices=m.vertices, faces=m.faces,
                             origin_above_bottom=ORIGIN_ABOVE_BOTTOM, hue=_hue())


def FakeCamera(head, truth_xy, spec):  # noqa: N802 - 테스트 가독성용 짧은 팩토리
    return fakes.FakeCamera(head, truth_xy, spec, BLUE_RGB, TABLE_Z)


FakeHead = fakes.FakeHead


def test_blob_init_xy_recovers_position_without_any_prior():
    spec, truth = _spec(), (0.22, 0.06)
    head = FakeHead()
    head.move(0.0, -20.0)
    frame = FakeCamera(head, truth, spec).capture()[0]
    xy = locate.blob_init_xy(frame.rgb, frame.k, spec.hue, head_fk.t_base_cam(0.0, -20.0), TABLE_Z + ORIGIN_ABOVE_BOTTOM)
    assert xy == pytest.approx(truth, abs=0.02)


def test_blob_init_xy_returns_none_when_nothing_matches_the_hue():
    blank = np.full((480, 640, 3), 20, np.uint8)
    assert locate.blob_init_xy(blank, K, _hue(), head_fk.t_base_cam(0.0, -20.0), 0.27) is None


def test_plan_poses_for_right_side_prefers_pans_that_see_negative_y_and_respects_tilt_limit():
    poses = locate.plan_poses(side="right", tilt_deg=-20.0)
    assert poses and all(t == -20.0 for _, t in poses)
    # 오른쪽(y<0) 테이블이 화면 중앙에 가까운 자세가 먼저 온다 — 첫 자세와 마지막 자세의 pan 이 달라야 의미가 있다.
    assert len({p for p, _ in poses}) == len(poses)
    t_first = head_fk.t_base_cam(*poses[0])
    forward = t_first[:3, 2]  # 카메라 광축(base 기준)
    assert forward[1] < 0.15, "오른쪽을 찾는데 첫 자세가 왼쪽을 본다"


def test_locate_finds_object_and_returns_home():
    spec, truth = _spec(), (0.22, -0.12)
    head = FakeHead()
    result = locate.locate_object(locate.LocateRequest(object_name="blue_cup", side="right"), spec, head,
                                  FakeCamera(head, truth, spec), table_z=TABLE_Z)
    assert result.found
    assert (result.x, result.y) == pytest.approx(truth, abs=3e-3)
    assert result.z == pytest.approx(TABLE_Z + ORIGIN_ABOVE_BOTTOM)
    assert result.n_used >= 1 and result.shots
    assert head.homed == 1


def test_locate_stops_early_once_enough_consistent_observations():
    spec, truth = _spec(), (0.22, -0.12)
    head = FakeHead()
    request = locate.LocateRequest(object_name="blue_cup", side="right", max_poses=6, min_used=2)
    result = locate.locate_object(request, spec, head, FakeCamera(head, truth, spec), table_z=TABLE_Z)
    assert result.found and len(head.moves) < 6


def test_locate_reports_not_found_without_inventing_a_value():
    spec = _spec()
    head = FakeHead()
    far_away = (5.0, 5.0)  # 어느 자세에서도 안 보인다
    result = locate.locate_object(locate.LocateRequest(object_name="blue_cup", side="right", max_poses=3), spec, head,
                                  FakeCamera(head, far_away, spec), table_z=TABLE_Z)
    assert not result.found
    assert result.x is None and result.y is None
    assert head.homed == 1


def test_locate_returns_home_even_when_the_head_fails_midway():
    spec = _spec()
    head = FakeHead(fail_on_move=1)
    result = locate.locate_object(locate.LocateRequest(object_name="blue_cup", side="right"), spec, head,
                                  FakeCamera(head, (0.22, -0.12), spec), table_z=TABLE_Z)
    assert not result.found and "목 이동 실패" in result.reason
    assert head.homed == 1


def test_locate_rejects_pose_above_tilt_limit_before_moving():
    spec = _spec()
    head = FakeHead()
    with pytest.raises(ValueError):
        locate.locate_object(locate.LocateRequest(object_name="blue_cup", poses=[(0.0, 0.0)]), spec, head,
                             FakeCamera(head, (0.22, 0.0), spec), table_z=TABLE_Z)
    assert head.moves == []


def test_find_head_commanders_matches_scripts_not_substrings_in_arguments():
    from perception_plus_plus_core.head_scan import adapters

    lines = [
        "  100 /home/user/rl_ws/sim2real/.venv/bin/python /home/user/rl_ws/sim2real/scripts/head_home.py --execute",
        "  101 python3 scripts/survey_head_poses.py --execute",
        "  102 /bin/bash -c echo locate_object head_home.py in a quoted string",
        "  103 /usr/bin/python3 pd_node --ros-args",
        "  104 python3 scripts/locate_object.py blue_cup --execute",
    ]
    found = adapters.find_head_commanders(lines, own_pid=104)
    assert [ln.split()[0] for ln in found] == ["100", "101"]


def test_locate_reports_stages_in_order_and_ends_with_returning():
    spec = _spec()
    head = FakeHead()
    stages = []
    locate.locate_object(locate.LocateRequest(object_name="blue_cup", side="right", max_poses=1), spec, head,
                         FakeCamera(head, (0.22, -0.12), spec), table_z=TABLE_Z, on_stage=lambda stage, pan, tilt: stages.append(stage))
    assert stages[0] == "planning"
    assert stages.index("moving") < stages.index("shooting") < stages.index("fitting") < stages.index("returning")
    assert stages[-1] == "returning"


def test_locate_stops_and_goes_home_when_cancelled():
    spec = _spec()
    head = FakeHead()
    result = locate.locate_object(locate.LocateRequest(object_name="blue_cup", side="right", max_poses=5), spec, head,
                                  FakeCamera(head, (5.0, 5.0), spec), table_z=TABLE_Z, should_stop=lambda: len(head.moves) >= 1)
    assert not result.found and "취소" in result.reason
    assert len(head.moves) == 1 and head.homed == 1


def test_repeatability_separates_frame_noise_from_return_to_pose_scatter():
    from perception_plus_plus_core.head_scan import repeatability

    spec, truth = _spec(), (0.22, -0.12)
    head = FakeHead()
    report = repeatability.run(head, FakeCamera(head, truth, spec), spec, pose=(-15.0, -20.0), detour=(15.0, -20.0),
                               n_visits=3, n_frames=2, table_z=TABLE_Z)
    assert len(report.visits) == 3 and all(len(v.fits) == 2 for v in report.visits)
    assert head.moves == [(-15.0, -20.0), (15.0, -20.0), (-15.0, -20.0), (15.0, -20.0), (-15.0, -20.0)]
    assert report.frame_noise_m < 1e-3 and report.visit_scatter_m < 1e-3
    assert report.angle_scatter_deg == pytest.approx(0.0, abs=1e-9)
    assert head.homed == 1
