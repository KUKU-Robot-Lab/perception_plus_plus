"""FK 잔여 오차가 상수 hand-eye 보정으로 줄어드는지 — 자세 1개를 빼고 맞춘 뒤 뺀 자세로 검증(교차 검증).
사용: python3 scripts/analysis/bolt_loo_calibration.py <서베이 폴더>"""
import json, sys, math, itertools
from pathlib import Path
import numpy as np, cv2
from scipy.optimize import least_squares
sys.path[:0] = ["/home/user/rl_ws/sim2real/scripts", "/home/user/rl_ws/sim2real/scripts/calib", "/home/user/rl_ws/perception_plus_plus"]
import table_cad_extrinsics as T
from perception_plus_plus_core.head_scan import head_fk
D = Path(sys.argv[1]); holes = T.cad_holes(); data = []
def backproject(Tbc, K, uv, zp):
    d = np.c_[(uv[:,0]-K[0,2])/K[0,0], (uv[:,1]-K[1,2])/K[1,1], np.ones(len(uv))]
    dr = d @ Tbc[:3,:3].T; o = Tbc[:3,3]; return o + dr*((zp-o[2])/dr[:,2])[:,None]
for r in json.load(open(D/"survey.json"))["records"]:
    z = np.load(D/r["frame"]); gray = cv2.cvtColor(z["rgb"], cv2.COLOR_RGB2GRAY); K = z["K"]
    TB = head_fk.t_base_cam(*r["measured"]); Tc, used, img, res, _ = T.solve(holes, gray, K, TB)
    data.append((TB, K, img, np.array([holes[i].xyz for i in used])))
def corr(x):
    Tc = np.eye(4); Tc[:3,:3] = cv2.Rodrigues(x[:3])[0]; Tc[:3,3] = x[3:]; return Tc
def resid(x, subset):
    out = []
    for i in subset:
        TB, K, img, obj = data[i]; e = backproject(TB @ corr(x), K, img, T.TABLE_TOP_Z)[:,:2] - obj[:,:2]; out.append(e.ravel()*1000)
    return np.concatenate(out)
def stat(x, i):
    n = np.linalg.norm(resid(x, [i]).reshape(-1,2), axis=1); return n.mean(), n.max()
x0 = np.zeros(6)
print("보정 전 평균/최대 오차 mm:", [tuple(round(v,1) for v in stat(x0,i)) for i in range(3)])
for test in range(3):
    train = [i for i in range(3) if i != test]
    x = least_squares(resid, x0, args=(train,)).x
    print(f"학습 자세 {train} → 검증 자세 {test}: 보정 전 {stat(x0,test)[0]:.1f} → 후 평균 {stat(x,test)[0]:.1f} 최대 {stat(x,test)[1]:.1f} mm · 보정 회전 {np.round(np.degrees(x[:3]),2).tolist()}° 이동 {np.round(x[3:]*1000,1).tolist()} mm")
x = least_squares(resid, x0, args=([0,1,2],)).x
print("3자세 전체 맞춤:", [tuple(round(v,1) for v in stat(x,i)) for i in range(3)], "회전", np.round(np.degrees(x[:3]),2).tolist(), "이동 mm", np.round(x[3:]*1000,1).tolist())
