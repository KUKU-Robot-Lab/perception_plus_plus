"""서베이 프레임들에서 홀더 패드를 찾아 홀더 위치(x, y)를 군집별로 보고한다. 사용: holder_pad_survey.py <서베이 폴더> [...]
같은 y(2 cm 안)로 3번 이상 잡힌 군집만 홀더로 본다. 640x480 에서는 패드가 어두워 검출이 들쭉날쭉하다."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["/home/user/rl_ws/perception_plus_plus"]
from perception_plus_plus_core.head_scan import head_fk, holder_pad as hp  # noqa: E402

CAND = Path("/home/user/rl_ws/perception_plus_plus/config/robot_camera_extrinsics/head_extrinsics_bolt_v3_candidate.yaml")
hits = []
for d in map(Path, sys.argv[1:]):
    for r in json.load(open(d / "survey.json"))["records"]:
        if r["frame"] is None:
            continue
        z = np.load(d / r["frame"])
        t_bc = head_fk.t_base_cam(*r["measured"], CAND)
        hits += [(r["measured"][0], hp.holder_xy_from_pad(p, t_bc, z["K"])) for p in hp.detect_pads(z["rgb"], t_bc, z["K"])]
clusters = []
for pan, xy in sorted(hits, key=lambda h: h[1][1]):
    if clusters and abs(xy[1] - np.mean([c[1] for c in clusters[-1]], axis=0)[1]) < 0.02 if clusters else False:
        clusters[-1].append((pan, xy))
    else:
        clusters.append([(pan, xy)])
for c in clusters:
    pts = np.array([xy for _, xy in c])
    tag = "홀더 후보" if len(c) >= 3 else "단발(무시)"
    print(f"{tag}: 관측 {len(c)}개 · 평균 x {pts[:, 0].mean():.4f} y {pts[:, 1].mean():.4f} · 표준편차 {np.round(1000 * pts.std(0), 1)} mm · pan {sorted(round(p) for p, _ in c)}")
