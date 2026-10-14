import cv2
import numpy as np
import pytest
import trimesh

from perception_plus_plus_core.head_scan import fakes, head_fk, locate, silhouette_fit as sf

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


def test_plan_poses_covers_the_whole_requested_side_including_the_far_edge():
    import numpy as np

    from perception_plus_plus_core.head_scan import scan_poses

    poses = locate.plan_poses(side="right")
    assert poses and len(set(poses)) == len(poses)
    assert all(tilt <= -10.0 for _, tilt in poses)
    points = scan_poses.table_points()
    right = points[points[:, 1] < 0]
    cam = scan_poses.default_camera()
    covered = np.zeros(len(right), dtype=bool)
    for pan, tilt in poses:
        covered |= scan_poses.visible_mask(right, head_fk.t_base_cam(pan, tilt), cam)
    assert covered.mean() >= 0.99, "오른쪽 가장자리까지 덮지 못한다"
    forward = head_fk.t_base_cam(*poses[0])[:3, 2]
    assert forward[1] < 0.15, "오른쪽을 찾는데 첫 자세가 왼쪽을 본다"


def test_plan_poses_left_side_looks_left_first():
    poses = locate.plan_poses(side="left")
    assert head_fk.t_base_cam(*poses[0])[:3, 2][1] > -0.15


def test_locate_finds_object_and_returns_home():
    spec, truth = _spec(), (0.22, -0.12)
    head = FakeHead()
    result = locate.locate_object(locate.LocateRequest(object_name="blue_cup", side="right"), spec, head,
                                  FakeCamera(head, truth, spec), table_z=TABLE_Z)
    assert result.found
    assert (result.x, result.y) == pytest.approx(truth, abs=3e-3)
    assert result.z == pytest.approx(TABLE_Z + ORIGIN_ABOVE_BOTTOM)
    assert result.n_used >= 1 and result.shots
    assert result.consistent == (result.n_used >= 2)
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


def _two_blob_frame(head, spec, target_xy, distractor_xy, distractor_radius):
    """같은 hue 의 컵(목표)과 더 굵은 원통(방해물)이 같이 서 있는 영상."""
    t_bc = head_fk.t_base_cam(*head.moves[-1])
    cup = sf.render_silhouette(spec.vertices + np.r_[target_xy, TABLE_Z + ORIGIN_ABOVE_BOTTOM], spec.faces, t_bc, fakes.DEFAULT_K, (480, 640))
    big = trimesh.creation.cylinder(radius=distractor_radius, height=HEIGHT, sections=48)
    other = sf.render_silhouette(big.vertices + np.r_[distractor_xy, TABLE_Z + ORIGIN_ABOVE_BOTTOM], big.faces, t_bc, fakes.DEFAULT_K, (480, 640))
    rgb = np.full((480, 640, 3), 20, np.uint8)
    rgb[cup | other] = BLUE_RGB
    return rgb


def test_best_blob_picks_the_blob_whose_shape_matches_not_the_largest():
    spec, target, distractor = _spec(), (0.22, -0.05), (0.22, 0.10)
    head = FakeHead()
    head.move(0.0, -20.0)
    rgb = _two_blob_frame(head, spec, target, distractor, distractor_radius=0.055)
    t_bc = head_fk.t_base_cam(0.0, -20.0)
    z = TABLE_Z + ORIGIN_ABOVE_BOTTOM
    largest = locate.blob_init_xy(rgb, fakes.DEFAULT_K, spec.hue, t_bc, z)
    assert largest == pytest.approx(distractor, abs=0.03), "전제: 가장 큰 덩어리는 방해물이다"
    found = locate.best_blob_init(spec, rgb, fakes.DEFAULT_K, t_bc, z)
    assert found is not None and found == pytest.approx(target, abs=0.01)


def test_best_blob_returns_none_when_no_blob_matches_the_shape():
    spec = _spec()
    head = FakeHead()
    head.move(0.0, -20.0)
    rgb = _two_blob_frame(head, spec, (5.0, 5.0), (0.22, 0.10), distractor_radius=0.09)
    assert locate.best_blob_init(spec, rgb, fakes.DEFAULT_K, head_fk.t_base_cam(0.0, -20.0), TABLE_Z + ORIGIN_ABOVE_BOTTOM) is None


def _profile(tmp_path, entries, strict=True):
    import yaml

    path = tmp_path / "prof.yaml"
    path.write_text(yaml.safe_dump({
        "profile": {"name": "t"}, "hand_eye": {"neck_to_camera": {"matrix": head_fk.load_neck_to_camera().tolist()}},
        "angle_tolerance_deg": 1.0, "strict": strict,
        "angles": [{"pan_deg": p, "tilt_deg": t, "bias_xy_m": list(b)} for p, t, b in entries]}), encoding="utf-8")
    from perception_plus_plus_core.head_scan import calib_profile as cpm
    return cpm.load_profile(path)


def test_estimate_adds_the_per_observation_bias_to_the_result():
    from perception_plus_plus_core.head_scan import standing_pose as sp

    spec = _spec()
    head = FakeHead()
    head.move(0.0, -20.0)
    frame = FakeCamera(head, (0.22, 0.05), spec).capture()[0]
    base = sp.StandingObs(frame.rgb, frame.k, 0.0, -20.0, (0.23, 0.04))
    shifted = sp.StandingObs(frame.rgb, frame.k, 0.0, -20.0, (0.23, 0.04), bias_xy=(0.002, -0.001))
    a = sp.estimate_standing([base], spec.vertices, spec.faces, spec.origin_above_bottom, spec.hue)
    b = sp.estimate_standing([shifted], spec.vertices, spec.faces, spec.origin_above_bottom, spec.hue)
    assert (b.x - a.x, b.y - a.y) == pytest.approx((0.002, -0.001), abs=1e-9)


def test_locate_with_profile_applies_bias_and_skips_uncalibrated_angles(tmp_path):
    spec, truth = _spec(), (0.22, -0.12)
    profile = _profile(tmp_path, [(-40.0, -20.0, (0.0, 0.0)), (-30.0, -20.0, (0.002, 0.0))])
    head = FakeHead()
    request = locate.LocateRequest(object_name="blue_cup", poses=[(-50.0, -20.0), (-40.0, -20.0), (-30.0, -20.0)], min_used=2)
    result = locate.locate_object(request, spec, head, FakeCamera(head, truth, spec), table_z=TABLE_Z, profile=profile)
    assert head.moves == [(-50.0, -20.0), (-40.0, -20.0), (-30.0, -20.0)]   # 이동은 하되
    assert [s.n_blobs for s in result.shots][0] == 0                       # 표에 없는 각도의 영상은 쓰지 않는다
    assert result.found and result.n_used == 2
    assert result.x == pytest.approx(truth[0] + 0.001, abs=2e-3)           # 두 각도 보정(0, +2 mm)의 가중 평균 근처


def test_plan_from_profile_only_returns_profile_angles(tmp_path):
    profile = _profile(tmp_path, [(-50.0, -30.0, (0.0, 0.0)), (-40.0, -20.0, (0.0, 0.0)), (45.0, -20.0, (0.0, 0.0))])
    poses = locate.plan_poses_from_profile(profile, side="right")
    assert set(poses) <= {(-50.0, -30.0), (-40.0, -20.0), (45.0, -20.0)}
    assert poses and poses[0] != (45.0, -20.0)
