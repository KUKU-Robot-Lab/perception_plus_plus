import numpy as np
from perception_plus_plus_ros import anchored_node as an


def test_cup_bbox_picks_highest_conf_cup():
    class _Box:
        def __init__(self, cls, conf, xyxy):
            self.cls = _T([cls]); self.conf = _T([conf]); self.xyxy = _T([xyxy])
    class _T:
        def __init__(self, v): self._v = np.asarray(v, float)
        def cpu(self): return self
        def numpy(self): return self._v
    class _Res:
        def __init__(self, boxes): self.boxes = boxes
    class _Boxes:
        def __init__(self, rows):
            self.cls = _T([r[0] for r in rows])
            self.conf = _T([r[1] for r in rows])
            self.xyxy = _T([r[2] for r in rows])
    class _Model:
        def __call__(self, rgb, verbose=False):
            return [_Res(_Boxes([(41, 0.9, [10, 10, 40, 40]),
                                 (41, 0.5, [50, 50, 60, 60]),
                                 (0, 0.99, [0, 0, 5, 5])]))]
    best = an.cup_bbox(_Model(), np.zeros((80, 80, 3), np.uint8), conf=0.25, class_id=41)
    assert best is not None
    assert abs(best[0] - 0.9) < 1e-6                       # 최고 conf cup
    assert tuple(best[1]) == (10.0, 10.0, 40.0, 40.0)


def test_reanchor_uses_engine_without_reinit():
    calls = []
    class _Est:
        def register(self, **kw): calls.append("register"); return np.eye(4)
    class _Cutie:
        def initialize(self, rgb, d): calls.append("cutie")
    class _Kal:
        def initiate(self, arr): calls.append("kalman"); return (arr, np.eye(len(arr)))
    class _Eng:
        estimator = _Est(); cutie = _Cutie(); kalman = _Kal(); est_iter = 5
        mask = None; kf_mean = None; kf_covariance = None
        def get_pose_array(self, pose): return np.zeros(7)
    class _Adapter:
        engine = _Eng()
    rgb = np.zeros((48, 64, 3), np.uint8)
    depth = np.full((48, 64), 0.6, np.float32)
    K = np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]])
    pose, mask = an.reanchor(_Adapter(), rgb, depth, K,
                             np.ones((48, 64), bool), np.ones((48, 64), np.uint8))
    assert calls == ["register", "cutie", "kalman"]        # 재init(initialize) 호출 없음
    assert pose.shape == (4, 4)
