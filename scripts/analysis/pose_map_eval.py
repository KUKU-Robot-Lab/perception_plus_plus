"""자세별 아핀 보정 지도(CAD 볼트로 만든다)가 볼트와 무관한 보드 코너의 자세 간 일관성을 좋게 하는지 본다.

사용: python3 scripts/analysis/pose_map_eval.py <capture_poses 폴더> [--yaml 후보.yaml] [--surveys ...]
자세마다: 그 영상의 볼트(추정 위치 → CAD 위치)로 아핀(x, y 선형 + 이동)을 맞춘다. 보드 코너의 상판 평면 위치에 같은 아핀을 적용한다.
보드는 지도 만들 때 쓰지 않았으므로 독립 검증이다. 보드 B 는 볼트에서 먼 모서리라 지도가 바깥으로도 맞는지 본다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["/home/user/rl_ws/perception_plus_plus", "/home/user/rl_ws/perception_plus_plus/scripts/analysis"]
import cad_calibrate as C  # noqa: E402

from perception_plus_plus_core.head_scan import head_fk, refit  # noqa: E402
from perception_plus_plus_core.head_scan.board_data import load_detections  # noqa: E402

Z = C.T.TABLE_TOP_Z


def affine(est, tru):
    return np.linalg.lstsq(np.c_[est, np.ones(len(est))], tru, rcond=None)[0]


def similarity(est, tru, scale=True):
    """est → tru 의 닮음(또는 강체) 변환을 (2x3 회전·배율 행렬, 이동)으로. 반사 없음."""
    mu_e, mu_t = est.mean(axis=0), tru.mean(axis=0)
    a, b = est - mu_e, tru - mu_t
    u, sig, vt = np.linalg.svd(b.T @ a)
    rot = u @ np.diag([1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    s = float(np.trace(np.diag(sig) @ np.diag([1.0, np.sign(np.linalg.det(u @ vt))])) / np.sum(a ** 2)) if scale else 1.0
    return s * rot, mu_t - s * rot @ mu_e


def spread_mm(per_pose):
    """per_pose: [{(board, id): xy}] → 코너별 자세 평균에서 벗어난 RMS(mm), 코너가 둘 이상의 자세에서 보인 것만."""
    keys = {}
    for d in per_pose:
        for k, v in d.items():
            keys.setdefault(k, []).append(v)
    sq = [np.sum((np.array(v) - np.mean(v, axis=0)) ** 2, axis=1) for v in keys.values() if len(v) >= 2]
    return float(np.sqrt(np.mean(np.concatenate(sq))) * 1000)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--yaml", type=Path, default=head_fk.CONFIG_DIR / "head_extrinsics_cad_v6_candidate.yaml")
    ap.add_argument("--surveys", nargs="*", default=[])
    args = ap.parse_args()
    frames = C.load_frames([], [args.folder], args.yaml)
    by_pose = {(round(f["pan"], 3), round(f["tilt"], 3)): f for f in frames}
    dets = load_detections(args.folder, 6, args.yaml)
    raw, mapped, mapped_const, rigid, simil = {}, {}, {}, {}, {}
    for d in dets:
        pan, tilt = d["rec"]["measured"]
        f = by_pose.get((round(pan, 3), round(tilt, 3)))
        if f is None:
            continue
        bolt = refit.Detection(pan, tilt, "bolt", np.arange(len(f["img"])), f["img"], f["k"])
        est = refit._ground_xy(bolt, refit.Params(), Z, args.yaml)
        w = affine(est, f["obj"][:, :2])
        c = (f["obj"][:, :2] - est).mean(axis=0)
        det = refit.Detection(pan, tilt, d["board"], d["ids"], d["uv"], d["k"])
        xy = refit._ground_xy(det, refit.Params(), Z, args.yaml)
        key = d["rec"]["index"]
        rr, rt = similarity(est, f["obj"][:, :2], scale=False)
        sr, st = similarity(est, f["obj"][:, :2], scale=True)
        for name, store, val in (("raw", raw, xy), ("map", mapped, np.c_[xy, np.ones(len(xy))] @ w), ("const", mapped_const, xy + c),
                                 ("rigid", rigid, xy @ rr.T + rt), ("sim", simil, xy @ sr.T + st)):
            store.setdefault(key, {}).update({(d["board"], int(i)): v for i, v in zip(d["ids"], val)})
    for label, store in (("보정 없음", raw), ("자세별 상수 이동", mapped_const), ("자세별 강체(회전+이동)", rigid),
                         ("자세별 닮음(+배율)", simil), ("자세별 아핀 지도", mapped)):
        print(f"{label:>10}: 보드 코너 자세 간 일관성 RMS {spread_mm(list(store.values())):.2f} mm ({len(store)}자세)")
    for board in ("A", "B"):
        for label, store in (("보정 없음", raw), ("자세별 아핀 지도", mapped)):
            sub = [{k: v for k, v in d.items() if k[0] == board} for d in store.values()]
            print(f"   보드 {board} · {label}: {spread_mm([d for d in sub if d]):.2f} mm")


if __name__ == "__main__":
    main()
