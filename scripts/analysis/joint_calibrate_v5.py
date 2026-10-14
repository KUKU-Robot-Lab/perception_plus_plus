"""볼트(CAD 절대 위치) + ChArUco 보드 A·B(자세 간 일관성, 거리 방향 척도)를 함께 맞춰 hand-eye 보정을 구하고, 자세 하나씩 빼며 검증한다.

사용: python3 scripts/analysis/joint_calibrate_v5.py --bolts <서베이 폴더>... --boards <capture_poses 폴더> [--free rot trans angles focal] [--write 후보.yaml]
미지수: hand-eye 보정(회전 3 + 이동 3, `T_neck_cam(v2) ∘ corr`), 선택 항(angles: pan·tilt 영점, focal: 초점 배율), 보드마다 (x, y, yaw).
잔차(mm): 볼트 = 픽셀을 상판 평면에 얹은 위치와 CAD 구멍 위치의 차, 보드 = 코너를 보드 높이 평면에 얹은 위치와 강체 보드 모델의 차.
볼트가 절대 위치를, 보드가 서로 다른 자세·위치에서의 일관성을 잡는다. 보드는 A4 종이라 높이 = 테이블 상면(0.205)으로 둔다.
검증: 자세(프레임) 하나를 빼고 맞춘 뒤 뺀 프레임의 볼트 오차와 보드 코너 오차를 잰다. 값은 제안용이며 head_extrinsics.yaml 은 고치지 않는다.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path[:0] = ["/home/user/rl_ws/sim2real/scripts", "/home/user/rl_ws/sim2real/scripts/calib", "/home/user/rl_ws/perception_plus_plus",
                "/home/user/rl_ws/perception_plus_plus/scripts/analysis"]
import handeye_calibrate as H  # noqa: E402
import table_cad_extrinsics as T  # noqa: E402

from perception_plus_plus_core.head_scan import head_fk, refit  # noqa: E402
from perception_plus_plus_core.head_scan.board_data import load_detections  # noqa: E402

BOARD_Z = 0.205  # A4 종이 두께(약 0.1 mm)는 무시
V2 = head_fk.EXTRINSICS_YAML
MIN_BOLTS = H.MIN_BOLTS


def load_bolt_frames(survey_dirs, capture_dirs):
    frames, _ = H.load_frames(survey_dirs)
    holes = T.cad_holes()
    for d in map(Path, capture_dirs):
        for rec in json.load(open(d / "poses.json"))["records"]:
            z = np.load(d / f"pose_{rec['index']}_0.npz")
            pan, tilt = rec["measured"]
            try:
                _, used, img, _, _ = T.solve(holes, cv2.cvtColor(z["rgb"], cv2.COLOR_RGB2GRAY), z["K"], head_fk.t_base_cam(pan, tilt))
            except RuntimeError:
                continue
            if len(used) >= MIN_BOLTS:
                frames.append({"tag": f"{d.name}[{rec['index']}]", "pan": pan, "tilt": tilt, "k": z["K"], "img": img,
                               "obj": np.array([holes[i].xyz for i in used])})
    return frames


def unpack(x, free, names):
    i = 6
    params = dict(rot=x[:3], trans=x[3:6], pan_off_deg=0.0, tilt_off_deg=0.0, f_scale=1.0)
    if "angles" in free:
        params["pan_off_deg"], params["tilt_off_deg"], i = float(x[i]), float(x[i + 1]), i + 2
    if "focal" in free:
        params["f_scale"], i = 1.0 + float(x[i]), i + 1
    boards = {n: refit.BoardPose(float(x[i + 3 * j]), float(x[i + 3 * j + 1]), float(x[i + 3 * j + 2])) for j, n in enumerate(names)}
    return refit.Params(**params), boards


def residuals(x, bolts, dets, free, names, mirror, w_board=1.0):
    params, boards = unpack(x, free, names)
    out = []
    for f in bolts:
        det = refit.Detection(f["pan"], f["tilt"], "bolt", np.arange(len(f["img"])), f["img"], f["k"])
        out.append(((refit._ground_xy(det, params, T.TABLE_TOP_Z, V2) - f["obj"][:, :2]) * 1000).ravel())
    for d in dets:
        model = refit.board_points(boards[d.board], d.ids, BOARD_Z, mirror)[:, :2]
        out.append(w_board * ((refit._ground_xy(d, params, BOARD_Z, V2) - model) * 1000).ravel())
    return np.concatenate(out)


def fit(bolts, dets, free, names, mirror, board_init, start6, w_board=1.0):
    """start6: hand-eye 보정 시작값(회전 3, 이동 3) — 보드 시작 위치가 이 보정으로 만든 것이라 같이 시작해야 지역해를 피한다."""
    n_extra = (2 if "angles" in free else 0) + (1 if "focal" in free else 0)
    x0 = np.r_[start6, np.zeros(n_extra), np.ravel([[board_init[n].x, board_init[n].y, board_init[n].yaw] for n in names])]
    return least_squares(residuals, x0, args=(bolts, dets, free, names, mirror, w_board)).x


def rms(vec):
    return float(np.sqrt(np.mean(np.square(vec)))) if len(vec) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--bolts", nargs="+", required=True)
    ap.add_argument("--boards", nargs="+", required=True)
    ap.add_argument("--free", nargs="*", default=[])
    ap.add_argument("--write", default=None)
    ap.add_argument("--gate-mm", type=float, default=15.0, help="기준 후보(v4) 오차가 이보다 큰 볼트 프레임은 짝짓기 실패로 보고 뺀다")
    ap.add_argument("--weight-board", type=float, default=1.0)
    args = ap.parse_args()
    free = tuple(args.free)

    bolts = load_bolt_frames(args.bolts, args.boards)
    gate_yaml = head_fk.CONFIG_DIR / "head_extrinsics_joint_v4_candidate.yaml"
    kept = []
    for f in bolts:
        det = refit.Detection(f["pan"], f["tilt"], "bolt", np.arange(len(f["img"])), f["img"], f["k"])
        err = np.linalg.norm((refit._ground_xy(det, refit.Params(), T.TABLE_TOP_Z, gate_yaml) - f["obj"][:, :2]) * 1000, axis=1).mean()
        if err > args.gate_mm:
            print(f"  볼트 프레임 {f['tag']} 제외: 기준 후보 오차 {err:.0f} mm (볼트 짝짓기 실패로 판단, tilt {f['tilt']:+.1f})")
        else:
            kept.append(f)
    bolts = kept
    dets = []
    for d in args.boards:
        dets += [refit.Detection(pan=e["rec"]["measured"][0], tilt=e["rec"]["measured"][1], board=e["board"], ids=e["ids"], uv=e["uv"], k=e["k"])
                 for e in load_detections(Path(d), 6, V2)]
    names = sorted({d.board for d in dets})
    base = refit.fit(dets, V2, BOARD_Z, ("rot", "trans"))
    mirror, board_init = base.mirror, base.boards
    print(f"볼트 프레임 {len(bolts)}개 · 보드 관측 {len(dets)}개 · 풀 항목 rot trans {' '.join(free)} · 보드 거울상 {mirror}")

    start6 = np.r_[base.params.rot, base.params.trans]
    w = args.weight_board
    x_full = fit(bolts, dets, free, names, mirror, board_init, start6, w)
    params, boards = unpack(x_full, free, names)
    nb = sum(len(f["img"]) * 2 for f in bolts)
    res = residuals(x_full, bolts, dets, free, names, mirror, 1.0)
    print(f"전체 맞춤: 볼트 RMS {rms(res[:nb]) / np.sqrt(2):.2f} mm/축→2D {rms(res[:nb]) * np.sqrt(2):.2f} mm · 보드 코너 RMS 2D {rms(res[nb:]) * np.sqrt(2):.2f} mm")
    print(f"  회전 mrad {np.round(params.rot * 1000, 2).tolist()} · 이동 mm {np.round(params.trans * 1000, 2).tolist()} · "
          f"pan/tilt 영점 {params.pan_off_deg:+.3f}/{params.tilt_off_deg:+.3f}° · 초점 배율 {params.f_scale:.4f}")

    held_bolt, held_board = [], []
    tags = [f["tag"] for f in bolts]
    poses = sorted({(d.pan, d.tilt) for d in dets})
    for i, f in enumerate(bolts):  # 볼트 프레임 하나 빼기
        rest = bolts[:i] + bolts[i + 1:]
        x = fit(rest, dets, free, names, mirror, boards, x_full[:6], w)
        r = residuals(x, [f], [], free, names, mirror)
        held_bolt.append(np.linalg.norm(r.reshape(-1, 2), axis=1).mean())
        print(f"  볼트 {f['tag']:>22} pan {f['pan']:+7.2f} tilt {f['tilt']:+6.2f}: 빼고 맞춘 뒤 오차 {held_bolt[-1]:.2f} mm")
    for pose in poses:  # 보드 자세 하나 빼기(그 자세의 볼트 프레임도 같이 뺀다)
        rest_dets = [d for d in dets if (d.pan, d.tilt) != pose]
        rest_bolts = [f for f in bolts if (round(f["pan"], 3), round(f["tilt"], 3)) != (round(pose[0], 3), round(pose[1], 3))]
        x = fit(rest_bolts, rest_dets, free, names, mirror, boards, x_full[:6], w)
        r = residuals(x, [], [d for d in dets if (d.pan, d.tilt) == pose], free, names, mirror, 1.0)
        held_board.append(rms(r) * np.sqrt(2))
        print(f"  보드 pan {pose[0]:+7.2f} tilt {pose[1]:+6.2f}: 빼고 맞춘 뒤 코너 오차 {held_board[-1]:.2f} mm ({sum(1 for d in dets if (d.pan, d.tilt) == pose)}장)")
    print(f"교차검증(하나 빼고): 볼트 평균 {np.mean(held_bolt):.2f} mm (최대 {max(held_bolt):.2f}, {len(tags)}프레임) · "
          f"보드 코너 RMS {np.sqrt(np.mean(np.square(held_board))):.2f} mm (최대 {max(held_board):.2f}, {len(poses)}자세)")
    if args.write:
        H.write_candidate(args.write, x_full[:6], f"볼트 {len(bolts)}프레임 + 보드 A·B {len(poses)}자세 결합 맞춤(v5), 교차검증 볼트 {np.mean(held_bolt):.2f} mm · 보드 {np.sqrt(np.mean(np.square(held_board))):.2f} mm")
        print("후보 파일 기록:", args.write)


if __name__ == "__main__":
    main()
