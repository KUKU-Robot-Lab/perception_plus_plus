import cv2
import numpy as np
import pytest
import trimesh

from perception_plus_plus_core.head_scan import head_fk, silhouette_fit as sf, standing_pose as sp

K = np.array([[606.6, 0, 320.0], [0, 605.7, 240.6], [0, 0, 1.0]])
HEIGHT = 0.13
ORIGIN_ABOVE_BOTTOM = HEIGHT / 2
BLUE_RGB = (30, 60, 220)  # OpenCV hue 약 118


def _mesh():
    m = trimesh.creation.cylinder(radius=0.0354, height=HEIGHT, sections=48)
    return m.vertices, m.faces


def _obs(pan, truth_xy, init_offset=(0.01, -0.008)):
    verts, faces = _mesh()
    t_bc = head_fk.t_base_cam(pan, -20.0)
    sil = sf.render_silhouette(verts + np.r_[truth_xy, sp.TABLE_TOP_Z + ORIGIN_ABOVE_BOTTOM], faces, t_bc, K, (480, 640))
    rgb = np.full((480, 640, 3), 20, np.uint8)
    rgb[sil] = BLUE_RGB
    return sp.StandingObs(rgb, K, pan, -20.0, (truth_xy[0] + init_offset[0], truth_xy[1] + init_offset[1]))


def _hue():
    return float(cv2.cvtColor(np.array([[BLUE_RGB]], np.uint8), cv2.COLOR_RGB2HSV)[0, 0, 0])


def test_estimate_standing_recovers_position_and_fixes_z():
    truth = (0.22, 0.06)
    verts, faces = _mesh()
    est = sp.estimate_standing([_obs(p, truth) for p in (-30.0, 0.0, 30.0)], verts, faces, ORIGIN_ABOVE_BOTTOM, _hue())
    assert est is not None and est.n_used == 3
    assert (est.x, est.y) == pytest.approx(truth, abs=2e-3)
    assert est.z == pytest.approx(0.205 + ORIGIN_ABOVE_BOTTOM)
    assert est.spread_m < 3e-3


def test_estimate_standing_drops_outlier_observation():
    truth = (0.22, 0.06)
    verts, faces = _mesh()
    observations = [_obs(p, truth) for p in (-30.0, 0.0, 30.0)] + [_obs(10.0, (0.25, 0.06), init_offset=(0, 0))]
    est = sp.estimate_standing(observations, verts, faces, ORIGIN_ABOVE_BOTTOM, _hue())
    assert est.n_rejected == 1
    assert (est.x, est.y) == pytest.approx(truth, abs=2e-3)


def test_estimate_standing_returns_none_without_a_usable_mask():
    verts, faces = _mesh()
    blank = sp.StandingObs(np.full((480, 640, 3), 20, np.uint8), K, 0.0, -20.0, (0.22, 0.06))
    assert sp.estimate_standing([blank], verts, faces, ORIGIN_ABOVE_BOTTOM, _hue()) is None


HOLDER_ORIGIN_Z = 0.235  # 받침 윗면(sim2real cup_holders.yaml). 컵 바닥은 이보다 25 mm 아래.
HOLDER_RING_RADIUS = 0.040
HOLDER_RING_BOTTOM, HOLDER_RING_TOP = -0.030, 0.030
CUP_BOTTOM_IN_HOLDER = HOLDER_ORIGIN_Z - 0.025


def _holder_ring():
    m = trimesh.creation.cylinder(radius=HOLDER_RING_RADIUS, height=HOLDER_RING_TOP - HOLDER_RING_BOTTOM, sections=48)
    return m.vertices + np.array([0, 0, (HOLDER_RING_TOP + HOLDER_RING_BOTTOM) / 2]), m.faces


def _obs_in_holder(pan, truth_xy, init_offset=(0.01, -0.008)):
    """홀더 링이 컵 아랫부분을 가린 영상(링이 앞에 있으니 링 실루엣 안의 컵 화소는 지운다)."""
    verts, faces = _mesh()
    t_bc = head_fk.t_base_cam(pan, -20.0)
    cup = sf.render_silhouette(verts + np.r_[truth_xy, CUP_BOTTOM_IN_HOLDER + ORIGIN_ABOVE_BOTTOM], faces, t_bc, K, (480, 640))
    ring_v, ring_f = _holder_ring()
    ring = sf.render_silhouette(ring_v + np.r_[truth_xy, HOLDER_ORIGIN_Z], ring_f, t_bc, K, (480, 640))
    rgb = np.full((480, 640, 3), 20, np.uint8)
    rgb[cup & ~ring] = BLUE_RGB
    return sp.StandingObs(rgb, K, pan, -20.0, (truth_xy[0] + init_offset[0], truth_xy[1] + init_offset[1]))


def test_estimate_standing_in_holder_uses_given_base_z_and_ignores_occluded_part():
    truth = (0.38, 0.07)
    verts, faces = _mesh()
    ring_v, ring_f = _holder_ring()
    est = sp.estimate_standing([_obs_in_holder(p, truth) for p in (-30.0, 0.0, 30.0)], verts, faces, ORIGIN_ABOVE_BOTTOM,
                               _hue(), base_z=CUP_BOTTOM_IN_HOLDER, occluder=(ring_v, ring_f, HOLDER_ORIGIN_Z))
    assert est is not None and est.n_used == 3
    assert (est.x, est.y) == pytest.approx(truth, abs=2e-3)
    assert est.z == pytest.approx(CUP_BOTTOM_IN_HOLDER + ORIGIN_ABOVE_BOTTOM)


def test_cache_does_not_refit_known_observations_and_gives_the_same_estimate(monkeypatch):
    """관측을 하나씩 늘려 부를 때 앞 관측을 다시 맞추지 않고, 결과는 캐시 없이와 같다."""
    truth = (0.22, 0.06)
    verts, faces = _mesh()
    observations = [_obs(p, truth) for p in (-30.0, 0.0, 30.0)]
    calls = {"n": 0}
    real = sf.fit_xy

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(sf, "fit_xy", counting)
    cache: dict = {}
    for n in (1, 2, 3):
        est = sp.estimate_standing(observations[:n], verts, faces, ORIGIN_ABOVE_BOTTOM, _hue(), cache=cache)
    assert calls["n"] == 3  # 1+1+1, 캐시가 없으면 1+2+3
    plain = sp.estimate_standing(observations, verts, faces, ORIGIN_ABOVE_BOTTOM, _hue())
    assert (est.x, est.y, est.n_used) == pytest.approx((plain.x, plain.y, plain.n_used))
