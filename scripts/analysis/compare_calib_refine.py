"""sim2real calib/refine_head_handeye.py 의 모델(hand-eye 6 + 보드 6자유도, 재투영 최소화)을 우리 캡처(board_1014a, 보드 A)에 돌려
v5·v6 와 hand-eye 가 같은지 비교한다. 기존 도구의 잔차 함수(`residual_fn`)를 그대로 불러 쓰고 검출만 우리 것(OpenCV 4.5 API)으로 바꿨다.

사용: python3 scripts/analysis/compare_calib_refine.py <capture_poses 폴더> [--write 후보.yaml]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path[:0] = ["/home/user/rl_ws/sim2real/scripts", "/home/user/rl_ws/sim2real/scripts/calib", "/home/user/rl_ws/perception_plus_plus",
                "/home/user/rl_ws/perception_plus_plus/scripts/analysis"]
import refine_head_handeye as R  # noqa: E402

from perception_plus_plus_core.head_scan import board_consistency as bc, head_fk  # noqa: E402
from perception_plus_plus_core.head_scan.board_data import load_detections  # noqa: E402


def board_objp(ids):
    ids = np.asarray(ids).astype(int)
    return np.column_stack([((ids % (bc.SX - 1)) + 1) * bc.SQUARE, ((ids // (bc.SX - 1)) + 1) * bc.SQUARE, np.zeros(len(ids))])


def delta(a, b):
    """두 hand-eye 행렬 사이의 (이동 mm, 회전 mrad)."""
    d = np.linalg.inv(a) @ b
    return float(np.linalg.norm(d[:3, 3]) * 1000), float(np.linalg.norm(cv2.Rodrigues(d[:3, :3])[0]) * 1000)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--write", default=None)
    args = ap.parse_args()
    dets = [d for d in load_detections(args.folder, 6, head_fk.EXTRINSICS_YAML) if d["board"] == "A"]
    obs = [{"pan": d["rec"]["measured"][0], "tilt": d["rec"]["measured"][1], "imgp": d["uv"], "objp": board_objp(d["ids"]), "K": d["k"]} for d in dets]
    # 시작값: v2 hand-eye + 첫 자세의 PnP 로 얻은 보드 위치
    t_nc0 = head_fk.load_neck_to_camera()
    o = obs[0]
    ok, rvec, tvec = cv2.solvePnP(o["objp"], o["imgp"], o["K"], None)
    pan_u, tilt_u = R.urdf_from_encoder(o["pan"], o["tilt"])
    t_bb0 = R.t_base_neck(pan_u, tilt_u) @ t_nc0 @ R.se3(rvec.ravel(), tvec.ravel())
    x0 = np.r_[R.se3_params(t_nc0), R.se3_params(t_bb0)]
    res = least_squares(R.residual_fn(obs), x0)
    t_nc = R.se3(res.x[0:3], res.x[3:6])
    print(f"기존 모델(보드 A {len(obs)}자세, 재투영 최소화): RMS {R.rms(res.fun):.3f} px")
    cands = {n: head_fk.load_neck_to_camera(head_fk.CONFIG_DIR / f) for n, f in (("v2", "head_extrinsics.yaml"), ("v3", "head_extrinsics_bolt_v3_candidate.yaml"),
             ("v4", "head_extrinsics_joint_v4_candidate.yaml"), ("v5", "head_extrinsics_joint_v5_candidate.yaml"), ("v6(CAD 볼트만)", "head_extrinsics_cad_v6_candidate.yaml"))}
    for n, m in cands.items():
        dt, dr = delta(t_nc, m)
        print(f"  기존 도구 결과 대 {n:>14}: 이동 차이 {dt:5.1f} mm · 회전 차이 {dr:5.1f} mrad")
    if args.write:
        rows = "\n".join("    - [" + ", ".join(f"{v:+.9f}" for v in row) + "]" for row in t_nc)
        Path(args.write).write_text(f"# hand-eye 후보 — sim2real refine_head_handeye 모델을 board_1014a 보드 A 에 돌린 값\nneck_to_camera:\n  matrix:\n{rows}\n", encoding="utf-8")
        print("후보 파일 기록:", args.write)


if __name__ == "__main__":
    main()
