import pytest

from perception_plus_plus_core.head_scan import ref_average as RA

IDENT = (1.0, 0.0, 0.0, 0.0)


def _sol(pos, quat=IDENT, res=2.0):
    return RA.Solution(position_m=pos, quat_wxyz=quat, res_px=res)


def test_identical_solutions_are_accepted_with_zero_spread():
    # Arrange
    sols = [_sol((0.06, 0.04, 0.8)) for _ in range(3)]

    # Act
    avg = RA.average(sols)

    # Assert
    assert avg.accepted and avg.reasons == ()
    assert avg.spread_mm == pytest.approx(0.0, abs=1e-9)
    assert avg.position_m == pytest.approx((0.06, 0.04, 0.8))


def test_spread_above_limit_is_rejected():
    # Arrange: 두 장이 y 로 12 mm 벗어나 평균에서 각 6 mm 떨어짐(기준 5 mm)
    sols = [_sol((0.0, 0.0, 0.8)), _sol((0.0, 0.012, 0.8))]

    # Act
    avg = RA.average(sols)

    # Assert
    assert not avg.accepted
    assert any("위치 차" in r for r in avg.reasons)


def test_residual_above_limit_is_rejected():
    # Arrange
    sols = [_sol((0.0, 0.0, 0.8), res=9.0) for _ in range(3)]

    # Act
    avg = RA.average(sols)

    # Assert
    assert not avg.accepted
    assert any("재투영" in r for r in avg.reasons)


def test_quaternion_sign_flip_is_treated_as_same_rotation():
    # Arrange: -q 는 같은 회전
    sols = [_sol((0.0, 0.0, 0.8), quat=(1.0, 0.0, 0.0, 0.0)), _sol((0.0, 0.0, 0.8), quat=(-1.0, 0.0, 0.0, 0.0))]

    # Act
    avg = RA.average(sols)

    # Assert
    assert avg.quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert avg.accepted


def test_empty_input_raises():
    with pytest.raises(ValueError):
        RA.average([])


def test_tick_stable_requires_equal_nonempty_dicts():
    assert RA.tick_stable({"pan": 2033, "tilt": 2864}, {"pan": 2033, "tilt": 2864})
    assert not RA.tick_stable({"pan": 2033, "tilt": 2864}, {"pan": 2034, "tilt": 2864})
    assert not RA.tick_stable({}, {})
