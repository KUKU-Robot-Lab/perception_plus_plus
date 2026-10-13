"""목 자세 서베이(head_scan.survey)의 순수 부분 검증 — 자세 검증, head_home 출력 파싱."""
from __future__ import annotations

import pytest

from perception_plus_plus_core.head_scan import survey

HEAD_HOME_OUT = """\
설정 /home/user/rl_ws/sim2real/config/head_home.yaml
  pan(id=1) → +15.00° (tick 2218)
  tilt(id=2) → -20.00° (tick 1820)
4s 정착 후 검증
  pan: 목표 +15.00° · 실제 +15.136° · 오차 +0.136° · σ 0.041° · I=400 · ✓
  tilt: 목표 -20.00° · 실제 -20.000° · 오차 +0.000° · σ 0.000° · I=400 · ✓
"""


def test_parse_head_home_reads_measured_mean_angles():
    parsed = survey.parse_head_home(HEAD_HOME_OUT)
    assert parsed.pan_deg == pytest.approx(15.136)
    assert parsed.tilt_deg == pytest.approx(-20.0)
    assert parsed.ok


def test_parse_head_home_flags_failed_verification():
    bad = HEAD_HOME_OUT.replace("I=400 · ✓", "I=400 · ❌", 1)
    assert not survey.parse_head_home(bad).ok


def test_parse_head_home_rejects_output_without_measurements():
    with pytest.raises(ValueError):
        survey.parse_head_home("아무 출력")


def test_validate_poses_accepts_any_pan_and_tilt_up_to_limit():
    survey.validate_poses([(15.0, -20.0), (-170.0, -40.0), (0.0, -10.0)], max_tilt_deg=-10.0)


def test_validate_poses_rejects_tilt_toward_gravity_beyond_limit():
    with pytest.raises(ValueError, match="tilt"):
        survey.validate_poses([(0.0, -9.9)], max_tilt_deg=-10.0)


def test_validate_poses_rejects_non_finite_and_empty():
    with pytest.raises(ValueError):
        survey.validate_poses([(float("nan"), -20.0)], max_tilt_deg=-10.0)
    with pytest.raises(ValueError):
        survey.validate_poses([], max_tilt_deg=-10.0)


def test_parse_pose_list_reads_semicolon_pairs():
    assert survey.parse_pose_list("15,-20;-15,-20; 0,-20") == [(15.0, -20.0), (-15.0, -20.0), (0.0, -20.0)]
    with pytest.raises(ValueError):
        survey.parse_pose_list("15;-20")


STILL_OFF = HEAD_HOME_OUT.replace("실제 +15.136° · 오차 +0.136°", "실제 +28.600° · 오차 -1.400°")
MOVING = HEAD_HOME_OUT.replace("σ 0.041°", "σ 0.800°")


def test_parse_head_home_reads_errors_and_sigmas():
    parsed = survey.parse_head_home(HEAD_HOME_OUT)
    assert parsed.pan_err_deg == pytest.approx(0.136)
    assert parsed.pan_sigma_deg == pytest.approx(0.041)


def test_accept_uses_measured_angle_when_still_and_close_even_if_verification_failed():
    off = survey.parse_head_home(STILL_OFF.replace("I=400 · ✓", "I=400 · ❌", 1))
    assert not off.ok
    assert survey.accept(off)  # 1.4° 어긋나도 정지해 있고 3° 안이면 실측값으로 계속한다


def test_accept_rejects_moving_or_far_off_head():
    assert not survey.accept(survey.parse_head_home(MOVING))
    far = HEAD_HOME_OUT.replace("실제 +15.136° · 오차 +0.136°", "실제 +10.000° · 오차 -5.000°")
    assert not survey.accept(survey.parse_head_home(far))


def test_drop_stale_removes_identical_raw_pose_seen_at_a_different_head_pose():
    rec = lambda i, pan, pos: {"index": i, "measured": [pan, -20.0], "raw_cam": {"cup": {"pos": pos, "quat_wxyz": [1, 0, 0, 0]}}}  # noqa: E731
    records = [rec(0, 0.1, [0.2, 0.0, 0.55]), rec(1, 15.0, [0.2, 0.0, 0.55]), rec(2, -15.0, [0.14, 0.0, 0.56])]
    kept = survey.drop_stale(records)
    assert [sorted(r["raw_cam"]) for r in kept] == [["cup"], [], ["cup"]]
    assert records[1]["raw_cam"]  # 원본 불변
