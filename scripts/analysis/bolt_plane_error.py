"""볼트(CAD 구멍) 기준 절대 오차: 검출 볼트 픽셀을 각 목 자세 FK/정적 변환으로 테이블 평면에 되쏘아 CAD 좌표와 비교한다.
사용: python3 scripts/analysis/bolt_plane_error.py <서베이 폴더>   (survey.json + frame_*.npz). sim2real calib 모듈을 경로로 불러 읽기만 한다."""
import json, sys, math
from pathlib import Path
import numpy as np, cv2
sys.path[:0] = ["/home/user/rl_ws/sim2real/scripts", "/home/user/rl_ws/sim2real/scripts/calib", "/home/user/rl_ws/perception_plus_plus"]
import table_cad_extrinsics as T
from cup_pose_relay import load_extrinsics
from perception_plus_plus_core.head_scan import head_fk
D = Path(sys.argv[1]); holes = T.cad_holes()
ext = load_extrinsics(Path("/home/user/rl_ws/sim2real/config/global_camera_extrinsics.yaml")); TA = T.T_from(ext.cam_pos, ext.cam_quat)
def backproject(Tbc, K, uv, z_plane):
    d = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))]
    dr = d @ Tbc[:3, :3].T; o = Tbc[:3, 3]
    s = (z_plane - o[2]) / dr[:, 2]
    return o + dr * s[:, None]
rows = []
for r in json.load(open(D/"survey.json"))["records"]:
    z = np.load(D/r["frame"]); gray = cv2.cvtColor(z["rgb"], cv2.COLOR_RGB2GRAY); K = z["K"]
    pan, tilt = r["measured"]; TB = head_fk.t_base_cam(pan, tilt)
    Tc, used, img, res, shift = T.solve(holes, gray, K, TB)
    obj = np.array([holes[i].xyz for i in used])
    line = f"[{r['index']}] pan {pan:+7.3f} · 기준점 {len(used)}"
    for name, Tm in (("FK(B)", TB), ("정적(A)", TA), ("볼트PnP", Tc)):
        e = backproject(Tm, K, img, T.TABLE_TOP_Z)[:, :2] - obj[:, :2]; n = np.linalg.norm(e, axis=1) * 1000
        line += f" | {name} 평균오차 {n.mean():.1f} 최대 {n.max():.1f} mm 평균벡터 {np.round(e.mean(0)*1000,1).tolist()}"
    print(line)
