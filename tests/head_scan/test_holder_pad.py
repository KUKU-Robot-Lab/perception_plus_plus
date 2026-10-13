import cv2
import numpy as np
import pytest

from perception_plus_plus_core.head_scan import head_fk, holder_pad as hp

K = np.array([[606.6, 0, 320.0], [0, 605.7, 240.6], [0, 0, 1.0]])


def _image_with_pad(y: float, t_bc: np.ndarray) -> np.ndarray:
    rgb = np.full((480, 640, 3), 15, np.uint8)
    uv = hp._project(t_bc, K, hp._pad_corners(y))
    u0, v0 = np.floor(uv.min(0)).astype(int)
    u1, v1 = np.ceil(uv.max(0)).astype(int)
    cv2.rectangle(rgb, (u0, v0), (u1, v1), (150, 150, 150), -1)  # 흰 패드(그늘)
    cv2.rectangle(rgb, (u0 + (u1 - u0) // 4, v0 + (v1 - v0) // 4), (u1 - (u1 - u0) // 4, v1 - (v1 - v0) // 4), (10, 10, 10), -1)  # 검은 마커
    return rgb


@pytest.mark.parametrize("pan", [0.0, 25.0, -25.0])
def test_detect_pads_recovers_holder_position_within_a_few_mm(pan):
    t_bc = head_fk.t_base_cam(pan, -20.0)
    truth_y = -0.05
    pads = hp.detect_pads(_image_with_pad(truth_y, t_bc), t_bc, K)
    assert len(pads) == 1
    xy = hp.holder_xy_from_pad(pads[0], t_bc, K)
    assert xy == pytest.approx([hp.HOLDER_X, truth_y], abs=5e-3)


def test_detect_pads_ignores_empty_dark_image():
    t_bc = head_fk.t_base_cam(0.0, -20.0)
    assert hp.detect_pads(np.full((480, 640, 3), 15, np.uint8), t_bc, K) == []
