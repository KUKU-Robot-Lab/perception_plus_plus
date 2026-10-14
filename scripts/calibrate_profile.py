#!/usr/bin/env python3
"""목 카메라 보정 프로필을 **명령 하나로** 만든다: 촬영 → hand-eye 맞춤 → 각도별 보정 → 기존 FP++ 좌표계 앵커 → 아는 z → 검증 → 프로필 파일.

    # 실기에서 처음부터(목이 움직인다 — 사용자 허락 뒤에만, 허락은 그 실행 한 번)
    python3 scripts/calibrate_profile.py --name myrobot_1015 --tilt -30 --pans="-60,-40,-20,0,20,40" --execute

    # 이미 찍은 서베이로(목 안 움직임)
    python3 scripts/calibrate_profile.py --name test --surveys <서베이 폴더>...

    # 계획만
    python3 scripts/calibrate_profile.py --name x --tilt -30 --pans="-60,-40,-20,0,20,40" --dry-run

순서(전부 이 한 스크립트가 한다):
  1. 자세 = 기준 자세(앵커용) + tilt 고정 pan 목록. 촬영은 `survey_head_poses.py`(영상 + FP++ 원시 pose)를 부른다. tilt 에 맞춘 camera extrinsics 를
     만들어 FP++ 스냅샷 노드에 준다(기준 자세 값을 주면 tilt 가 다를 때 후보가 버려진다).
  2. 시작 hand-eye: 기준 자세 영상에서 CAD 구멍 전체 이동 탐색(`table_cad_extrinsics.solve`)으로 카메라 자세를 풀어 hand-eye 시작값을 얻는다
     (환경마다 hand-eye 가 달라 12~60 px 어긋나도 잡힌다).
  3. 다른 자세는 그 시작값으로 구멍을 투영해 구멍마다 작은 창에서만 찾는다(`bolt_prior_detect`, 반복 구멍 오짝짓기 방지). 맞춤 → 재검출 → 재맞춤.
  4. 각도별 보정 = 명령 자세마다 CAD 와 볼트 위치의 평균 차이. 5. 앵커 = 기준 자세에서 기존 정적 camera 블록과 같게(강체 + dz).
  6. known z(상판 + 물체 원점 높이). 7. 프레임 하나씩 빼는 검증과 요약을 실행 폴더에 쓴다.
출력: `sim2real/config/calib_profiles/<name>.yaml` (이미 있으면 거부) + 실행 폴더(`report/survey/calib_<name>/`)의 report.md.
쓰는 법: `object_pose_node.py --head-joint-topic ... --calib-profile <name>` 와 `locate_object.py --calib-profile <name>`.
필요한 것: 로봇 PC 에 sim2real 이 있고 vision 호스트에서 FP++ 가 뜰 수 있어야 한다. 테이블 CAD(env_v1)와 같은 테이블이어야 한다.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE / "analysis")]
import anchor_profile as AP  # noqa: E402
import bolt_prior_detect as B  # noqa: E402
import make_camera_extrinsics_at as MCE  # noqa: E402

from perception_plus_plus_core.head_scan import adapters, calib_profile as cpm, head_fk, refit, survey  # noqa: E402

T = B.T
Z = T.TABLE_TOP_Z
REF = (0.0, -20.0)
MIN_BOLTS_FIT, MIN_BOLTS_BIAS = 5, 4
TABLE_TOP_M = 0.205  # 줄자 확정(sim2real docs/TABLE_DATUM_2026-09-05.md)


# ── 순수 계산 ──────────────────────────────────────────────────────────────
def rotvec_trans(matrix: np.ndarray) -> np.ndarray:
    return np.r_[cv2.Rodrigues(matrix[:3, :3])[0].ravel(), matrix[:3, 3]]


def corr_matrix(x: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = cv2.Rodrigues(np.asarray(x[:3], float))[0], x[3:6]
    return m


def corr_between(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    """target = base ∘ corr 를 만족하는 corr 의 (회전벡터, 이동) 6 개."""
    return rotvec_trans(np.linalg.inv(base) @ target)


def group_bias(frames: list[dict], neck_to_camera: np.ndarray, min_bolts: int = MIN_BOLTS_BIAS) -> dict[tuple[float, float], dict]:
    """명령 자세별로 CAD − 추정 위치의 평균(상판, xy) 을 낸다. frames: {commanded, pan, tilt, k, img, obj}."""
    groups: dict[tuple[float, float], list[np.ndarray]] = {}
    meta: dict[tuple[float, float], dict] = {}
    for f in frames:
        if len(f["img"]) < min_bolts:
            continue
        det = refit.Detection(f["pan"], f["tilt"], "bolt", np.arange(len(f["img"])), f["img"], f["k"])
        est = _ground(det, neck_to_camera)
        groups.setdefault(f["commanded"], []).append(f["obj"][:, :2] - est)
        meta.setdefault(f["commanded"], {"pan": [], "tilt": []})
        meta[f["commanded"]]["pan"].append(f["pan"])
        meta[f["commanded"]]["tilt"].append(f["tilt"])
    out = {}
    for key, diffs in groups.items():
        d = np.vstack(diffs)
        shift = d.mean(axis=0)
        out[key] = {"pan": float(np.mean(meta[key]["pan"])), "tilt": float(np.mean(meta[key]["tilt"])), "bias": shift, "n_frames": len(diffs),
                    "n_bolts": len(d), "resid_mm": float(np.linalg.norm(d - shift, axis=1).mean() * 1000)}
    return out


def _ground(det: refit.Detection, neck_to_camera: np.ndarray) -> np.ndarray:
    path = Path("/tmp") / "calibrate_profile_hand_eye.yaml"
    path.write_text(_yaml_matrix(neck_to_camera), encoding="utf-8")
    head_fk.load_neck_to_camera.cache_clear()
    return refit._ground_xy(det, refit.Params(), Z, path)


def _yaml_matrix(matrix: np.ndarray) -> str:
    rows = "\n".join("    - [" + ", ".join(f"{v:+.9f}" for v in row) + "]" for row in matrix)
    return f"neck_to_camera:\n  matrix:\n{rows}\n"


# ── 데이터 ────────────────────────────────────────────────────────────────
def load_records(surveys: list[Path]) -> list[dict]:
    out = []
    for d in surveys:
        for r in json.load(open(d / "survey.json"))["records"]:
            if r["frame"]:
                out.append({"dir": d, "commanded": tuple(float(v) for v in r["commanded"]), "pan": r["measured"][0], "tilt": r["measured"][1],
                            "path": d / r["frame"]})
    return out


def detect_all(records: list[dict], neck_to_camera: np.ndarray, holes) -> list[dict]:
    path = Path("/tmp") / "calibrate_profile_prior.yaml"
    path.write_text(_yaml_matrix(neck_to_camera), encoding="utf-8")
    head_fk.load_neck_to_camera.cache_clear()
    frames = []
    for r in records:
        z = np.load(r["path"])
        img, obj = B.detect(cv2.cvtColor(z["rgb"], cv2.COLOR_RGB2GRAY), z["K"], head_fk.t_base_cam(r["pan"], r["tilt"], path), holes)
        frames.append({**r, "k": z["K"], "img": img, "obj": obj})
    return frames


def bootstrap(records: list[dict], base: np.ndarray, holes, log) -> np.ndarray:
    """기준 자세 영상으로 hand-eye 시작값: CAD 구멍 전체 이동 탐색 + PnP 로 풀린 T_base_cam 에서 역산."""
    refs = [r for r in records if abs(r["pan"] - REF[0]) <= 1.0 and abs(r["tilt"] - REF[1]) <= 1.0]
    if not refs:
        raise RuntimeError(f"기준 자세(pan {REF[0]}, tilt {REF[1]}) 영상이 없다 — 시작 hand-eye 를 못 구한다. 자세 목록에 기준 자세를 넣을 것")
    results = []
    for r in refs:
        z = np.load(r["path"])
        t_init = head_fk.t_base_neck(r["pan"], r["tilt"]) @ base
        try:
            t_solved, used, _, res, _ = T.solve(holes, cv2.cvtColor(z["rgb"], cv2.COLOR_RGB2GRAY), z["K"], t_init)
        except RuntimeError as exc:
            log(f"  시작값 풀기 실패({r['path'].name}): {exc}")
            continue
        results.append((np.linalg.inv(head_fk.t_base_neck(r["pan"], r["tilt"])) @ t_solved, len(used), float(np.mean(res))))
        log(f"  기준 자세 {r['path'].parent.name}/{r['path'].name}: 구멍 {len(used)}개 재투영 {np.mean(res):.2f} px")
    if not results:
        raise RuntimeError("기준 자세 영상에서 CAD 구멍으로 카메라 자세를 못 풀었다 — 조명·테이블 배치·hand-eye 시작값을 볼 것")
    return max(results, key=lambda x: x[1])[0]


def fit_hand_eye(frames: list[dict], base: np.ndarray, start: np.ndarray) -> np.ndarray:
    usable = [f for f in frames if len(f["img"]) >= MIN_BOLTS_FIT]
    if len(usable) < 4:
        raise RuntimeError(f"볼트가 {MIN_BOLTS_FIT}개 이상 잡힌 프레임이 {len(usable)}개뿐이다(4개 이상 필요)")
    base_path = Path("/tmp") / "calibrate_profile_base.yaml"
    base_path.write_text(_yaml_matrix(base), encoding="utf-8")
    head_fk.load_neck_to_camera.cache_clear()

    def residual(x):
        params = refit.Params(rot=x[:3], trans=x[3:6])
        return np.concatenate([((refit._ground_xy(refit.Detection(f["pan"], f["tilt"], "bolt", np.arange(len(f["img"])), f["img"], f["k"]), params, Z, base_path)
                                - f["obj"][:, :2]) * 1000).ravel() for f in usable])

    sol = least_squares(residual, corr_between(base, start))
    return base @ corr_matrix(sol.x)


def loo_bolt_error(frames: list[dict], base: np.ndarray, hand_eye: np.ndarray) -> float:
    usable = [f for f in frames if len(f["img"]) >= MIN_BOLTS_FIT]
    errs = []
    for i, f in enumerate(usable):
        rest = usable[:i] + usable[i + 1:]
        he = fit_hand_eye(rest, base, hand_eye)
        det = refit.Detection(f["pan"], f["tilt"], "bolt", np.arange(len(f["img"])), f["img"], f["k"])
        errs.append(float(np.linalg.norm(_ground(det, he) - f["obj"][:, :2], axis=1).mean() * 1000))
    return float(np.mean(errs))


# ── 실행 ──────────────────────────────────────────────────────────────────
def capture(args, run_dir: Path, base: np.ndarray, log) -> Path:
    busy = adapters.busy_head_commanders()
    if busy:
        raise RuntimeError("목을 명령하는 다른 프로세스가 있다 — 중단: " + "; ".join(busy))
    run_dir.mkdir(parents=True, exist_ok=True)
    camera_yaml = run_dir / "camera_extrinsics_tilt.yaml"
    base_hand_eye = run_dir / "base_hand_eye.yaml"
    base_hand_eye.write_text(_yaml_matrix(base), encoding="utf-8")
    camera_yaml.write_text(MCE.render(MCE.DEFAULT_BASE.read_text(encoding="utf-8"), 0.0, args.tilt, base_hand_eye), encoding="utf-8")
    log(f"tilt {args.tilt:+g} 용 camera extrinsics 를 만들었다: {camera_yaml}")
    subprocess.run(["ssh", "-o", "BatchMode=yes", args.vision_host, "cd ~/rl_ws/sim2real && bash scripts/vision/fpp_down.sh all"], capture_output=True, timeout=60)
    poses = ";".join(f"{p:g},{t:g}" for p, t in [REF] + [(pan, args.tilt) for pan in args.pan_list])
    cmd = [sys.executable, str(HERE / "survey_head_poses.py"), f"--poses={poses}", "--camera-extrinsics", str(camera_yaml),
           "--out-dir", str(run_dir / "survey"), "--execute"]
    log("촬영: " + " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"서베이가 실패했다(rc {res.returncode}) — 목은 복귀를 시도했다. {run_dir / 'survey'} 를 확인")
    return run_dir / "survey"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--tilt", type=float, default=-30.0)
    ap.add_argument("--pans", default="-60,-40,-20,0,20,40", help="tilt 고정으로 찍을 pan 목록(deg)")
    ap.add_argument("--surveys", nargs="*", type=Path, default=None, help="이미 찍은 서베이 폴더들(주면 촬영 안 함)")
    ap.add_argument("--base-hand-eye", type=Path, default=head_fk.EXTRINSICS_YAML, help="이 환경의 현재 hand-eye(보정은 이것 대비)")
    ap.add_argument("--table-top", type=float, default=TABLE_TOP_M)
    ap.add_argument("--vision-host", default="vision-3090")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--no-loo", action="store_true", help="프레임 하나씩 빼는 검증을 건너뜀(빠르게)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true", help="목을 움직여 촬영한다(사용자 허락 뒤)")
    args = ap.parse_args()
    args.pan_list = [float(v) for v in args.pans.split(",")]
    run_dir = args.run_dir or Path.home() / "rl_ws/report/survey" / f"calib_{args.name}"
    out = cpm.PROFILE_DIR / f"{args.name}.yaml"
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    poses = [REF] + [(p, args.tilt) for p in args.pan_list]
    survey.validate_poses(poses + [REF])
    log(f"프로필 {args.name}: 자세 {poses} (첫 자세 = 앵커용 기준 자세) → {out}")
    if out.exists():
        print(f"❌ {out} 가 이미 있다 — 덮어쓰지 않는다. 다른 이름으로", file=sys.stderr)
        return 2
    if args.dry_run or (not args.surveys and not args.execute):
        print("[dry-run] 촬영은 목을 움직이므로 --execute (사용자 허락 뒤). 이미 찍은 데이터면 --surveys <폴더>...")
        return 0

    base = head_fk.load_neck_to_camera(args.base_hand_eye)
    holes = T.cad_holes()
    surveys = list(args.surveys) if args.surveys else [capture(args, run_dir, base, log)]
    records = load_records(surveys)
    log(f"영상 {len(records)}장 (서베이 {[s.name for s in surveys]})")

    log("1) 시작 hand-eye (기준 자세 CAD 구멍 풀기)")
    hand_eye = bootstrap(records, base, holes, log)
    for it in (1, 2):
        frames = detect_all(records, hand_eye, holes)
        hand_eye = fit_hand_eye(frames, base, hand_eye)
        usable = [f for f in frames if len(f["img"]) >= MIN_BOLTS_FIT]
        errs = [float(np.linalg.norm(_ground(refit.Detection(f["pan"], f["tilt"], "b", np.arange(len(f["img"])), f["img"], f["k"]), hand_eye) - f["obj"][:, :2], axis=1).mean() * 1000)
                for f in usable]
        log(f"2) 맞춤 {it}회: 볼트 프레임 {len(usable)}개, 볼트 평균 오차 {np.mean(errs):.2f} mm")
    frames = detect_all(records, hand_eye, holes)

    groups = group_bias(frames, hand_eye)
    entries = tuple(cpm.AngleEntry(g["pan"], g["tilt"], (float(g["bias"][0]), float(g["bias"][1])), g["n_frames"], g["n_bolts"], g["resid_mm"])
                    for _, g in sorted(groups.items()))
    if not entries:
        raise RuntimeError("각도별 보정을 만들 프레임이 없다(볼트 4개 이상 잡힌 프레임 0)")
    for _, g in sorted(groups.items()):
        log(f"3) pan {g['pan']:+7.2f} tilt {g['tilt']:+6.2f}: 볼트 {g['n_bolts']:3d}개 · 보정 ({1000 * g['bias'][0]:+.2f}, {1000 * g['bias'][1]:+.2f}) mm · 보정 후 잔차 {g['resid_mm']:.2f} mm")
    plain = cpm.CalibProfile(name=args.name, neck_to_camera=hand_eye, entries=entries, strict=True,
                             notes=f"calibrate_profile.py 로 한 번에 만듦. 서베이 {[s.name for s in surveys]}, tilt {args.tilt:+g} pan {args.pans}, 기준 hand-eye {args.base_hand_eye.name}")
    plain_path = (run_dir if run_dir.exists() else Path("/tmp")) / f"{args.name}_plain.yaml"
    plain_path.parent.mkdir(parents=True, exist_ok=True)
    cpm.save_profile(plain, plain_path)

    log("4) 기존 FP++ 좌표계 앵커")
    anchor, stats = AP.compute_anchor(plain, plain_path, surveys, REF[0], REF[1], log=log)
    final = dataclasses.replace(plain, anchor=anchor, known_z=cpm.KnownZ(args.table_top, 0.015), notes=plain.notes + " + 앵커 + 아는 z")

    if not args.no_loo:
        log(f"5) 검증(프레임 하나 빼고 hand-eye 다시 맞춤): 볼트 평균 오차 {loo_bolt_error(frames, base, hand_eye):.2f} mm")
    cpm.save_profile(final, out)
    log(f"프로필 기록: {out} (각도 {len(entries)}개, 앵커 잔차 {stats['fit_resid_mm']:.2f} mm, FP++ z 관측 {stats['n_z_obs']}개)")
    log(f"사용: object_pose_node.py --head-joint-topic /head/joint_states --calib-profile {args.name}")
    log(f"      locate_object.py <물체> --side right --tilt {args.tilt:g} --calib-profile {args.name} --execute")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.md").write_text("# 보정 프로필 실행 기록\n\n```\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
