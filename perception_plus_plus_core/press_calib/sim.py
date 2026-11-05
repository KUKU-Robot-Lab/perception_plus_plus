"""press_calib 가벼운 sim — 카메라 없이 IK · 도달성 · 닫힌 루프 · 잡음 민감도만 본다 (순수 numpy, 로봇 구동 없음).

단계
    1. HandModel        URDF 만으로 오른팔 + 손 링크 FK (mimic 포함). 손 자세: 검지 펴고 나머지 접기
    2. solve_press_ik   손끝을 목표점에 두고, 검지가 보드 법선 반대(아래)를 향하게 하는 DLS IK
    3. plan_session     코너별 접근점 · 접촉점 IK + 관절 한계 + 테이블 여유 검사 → 통과 / 탈락 이유
    4. closed_loop      sim 관절각 → FK → fit_board_pose → 정답과 비교, extrinsics 를 일부러 틀어 compare_with_camera 가 찾는지
    5. sensitivity      인코더 · 접촉 위치 잡음 Monte-Carlo → 보드 위치/회전 오차 분포

자기 충돌(팔 ↔ 몸)은 보지 않는다 — 테이블 여유만. 카메라 포함 확인은 sim_e2e 장면에서 한다.
좌표: base = URDF 루트 body_root. table_z · 보드 자세는 이 프레임으로 넘긴다.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import (ARM_JOINTS, BASE_LINK, DEFAULT_URDF, FINGER_JOINTS, N_CORNERS, TIP_LINK,
               _rpy, compare_with_camera, fit_board_pose, plan_press_targets)

PROXIMAL_LINK = "r_hl_index_1"  # 검지 방향 = 이 링크 원점 → 손끝
CURL = {"r_hj_middle_1": 1.4, "r_hj_ring_1": 1.4, "r_hj_pinky_1": 1.4,
        "r_hj_thumb_1": 1.0, "r_hj_thumb_2": 0.4}  # 나머지 손가락 접기 (rad, 한계 안)
CLEARANCE_LINKS = ("r_hl_base", "r_hl_palm_1", "r_hl_palm_2", "r_hl_palm_3", "r_hl_thumb_tip",
                   "r_hl_middle_tip", "r_hl_ring_tip", "r_hl_pinky_tip", "r_al_7")
# 접기 · 의자세 기본값 (팔꿈치 굽힘). IK 는 여러 시작점에서 돈다.
IK_SEEDS = ((0.0, 0.3, 0.0, 1.2, 0.0, 0.0, 0.0), (0.5, 0.2, 0.3, 1.5, 0.0, 0.3, 0.0),
            (-0.3, 0.5, -0.3, 1.0, 0.3, -0.3, 0.0), (0.8, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0))


def _rot_axis(axis: np.ndarray, a: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray
    revolute: bool
    mimic: tuple[str, float, float] | None
    limit: tuple[float, float] | None


class HandModel:
    """URDF 트리를 직접 걷는 FK. 관절값 dict 를 받아 임의 링크의 base 자세를 준다."""

    def __init__(self, urdf_path: Path | str = DEFAULT_URDF) -> None:
        root = ET.fromstring(Path(urdf_path).read_text())
        self._by_child: dict[str, _Joint] = {}
        for j in root.iter("joint"):
            o = j.find("origin")
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = _rpy(*(float(v) for v in o.get("rpy", "0 0 0").split()))
                T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            ax = j.find("axis")
            m = j.find("mimic")
            lim = j.find("limit")
            lo_s, hi_s = (lim.get("lower"), lim.get("upper")) if lim is not None else (None, None)
            self._by_child[j.find("child").get("link")] = _Joint(  # type: ignore[union-attr]
                j.get("name", ""), j.find("parent").get("link"), T,  # type: ignore[union-attr]
                np.array([float(v) for v in (ax.get("xyz", "1 0 0") if ax is not None else "1 0 0").split()]),
                j.get("type") in ("revolute", "continuous"),
                (m.get("joint", ""), float(m.get("multiplier", 1)), float(m.get("offset", 0))) if m is not None else None,
                (float(lo_s), float(hi_s)) if lo_s and hi_s else None)
        self.limits = {j.name: j.limit for j in self._by_child.values() if j.revolute and j.limit}

    def link_pose(self, link: str, q: dict[str, float]) -> np.ndarray:
        T = np.eye(4)
        while link != BASE_LINK:
            j = self._by_child[link]
            step = j.origin.copy()
            if j.revolute:
                a = q.get(j.name, 0.0) if j.mimic is None else j.mimic[1] * q.get(j.mimic[0], 0.0) + j.mimic[2]
                step[:3, :3] = step[:3, :3] @ _rot_axis(j.axis, a)
            T = step @ T
            link = j.parent
        return T

    def link_pos(self, link: str, q: dict[str, float]) -> np.ndarray:
        return self.link_pose(link, q)[:3, 3]


def hand_q(arm_q: Sequence[float] | np.ndarray, index_q: float = 0.0) -> dict[str, float]:
    q = dict(zip(ARM_JOINTS, map(float, arm_q)))
    q.update(CURL)
    q[FINGER_JOINTS[0]] = float(index_q)
    return q


@dataclass
class IKResult:
    arm_q: np.ndarray
    pos_err_m: float
    dir_err_deg: float
    ok: bool


def _residual(model: HandModel, arm_q: np.ndarray, target: np.ndarray, down: np.ndarray,
              finger_len: float, w_dir: float) -> np.ndarray:
    q = hand_q(arm_q)
    tip = model.link_pos(TIP_LINK, q)
    prox = model.link_pos(PROXIMAL_LINK, q)
    # 근위 링크가 손끝에서 법선 방향(위)으로 finger_len 만큼 있어야 검지가 아래를 향한다
    return np.concatenate([tip - target, w_dir * (prox - (target - down * finger_len))])


def solve_press_ik(model: HandModel, target: np.ndarray, down: np.ndarray,
                   seeds: Sequence[Sequence[float] | np.ndarray] = IK_SEEDS, iters: int = 200,
                   tol_m: float = 1e-4, max_dir_deg: float = 15.0, w_dir: float = 0.3,
                   q_ref: np.ndarray | None = None, k_ref: float = 0.5) -> IKResult:
    """손끝 → target, 검지 방향 → down (단위벡터, 보통 −법선). 관절 한계 안으로 자른다.

    6축 목표는 LM 으로 정확히 맞추고, q_ref 를 주면 남는 1 자유도(영공간)만 그 자세 쪽으로 보낸다
    (접근 → 접촉이 다른 해로 튀지 않게).
    """
    def res(qq: np.ndarray) -> np.ndarray:
        return _residual(model, qq, target, down, finger_len, w_dir)

    down = np.asarray(down, float) / np.linalg.norm(down)
    q0 = hand_q(np.zeros(7))
    finger_len = float(np.linalg.norm(model.link_pos(TIP_LINK, q0) - model.link_pos(PROXIMAL_LINK, q0)))
    lo = np.array([model.limits[n][0] for n in ARM_JOINTS])
    hi = np.array([model.limits[n][1] for n in ARM_JOINTS])
    best: IKResult | None = None
    for seed in seeds:
        q = np.clip(np.asarray(seed, float), lo, hi)
        lam = 1e-3
        r = res(q)
        for _ in range(iters):
            if np.linalg.norm(r[:3]) < tol_m * 0.1 and np.linalg.norm(r[3:6]) < tol_m:
                break
            J = np.empty((r.size, 7))
            for k in range(7):
                dq = np.zeros(7)
                dq[k] = 1e-7
                J[:, k] = (res(q + dq) - r) / 1e-7
            step = np.linalg.solve(J.T @ J + lam * np.eye(7), -J.T @ r)
            if q_ref is not None:  # 영공간 투영: 목표를 건드리지 않고 q_ref 쪽으로
                step = step + (np.eye(7) - np.linalg.pinv(J) @ J) @ (k_ref * (q_ref - q))
            qn = np.clip(q + np.clip(step, -0.3, 0.3), lo, hi)
            rn = res(qn)
            if rn @ rn < r @ r or np.linalg.norm(rn) < tol_m:  # LM: 줄면 받아들이고 λ 를 줄인다
                q, r, lam = qn, rn, max(lam * 0.3, 1e-9)
            else:
                lam *= 10
                if lam > 1e3:
                    break
        out = _evaluate(model, q, target, down)
        out.ok = out.pos_err_m < 1e-3 and out.dir_err_deg < max_dir_deg
        if best is None or (out.ok, -out.pos_err_m - 1e-3 * out.dir_err_deg) > (best.ok, -best.pos_err_m - 1e-3 * best.dir_err_deg):
            best = out
        if out.ok and out.pos_err_m < tol_m * 2:
            break
    assert best is not None
    return best


def _evaluate(model: HandModel, q: np.ndarray, target: np.ndarray, down: np.ndarray) -> IKResult:
    hq = hand_q(q)
    tip = model.link_pos(TIP_LINK, hq)
    d = tip - model.link_pos(PROXIMAL_LINK, hq)
    ang = np.degrees(np.arccos(np.clip(d @ down / np.linalg.norm(d), -1, 1)))
    return IKResult(q.copy(), float(np.linalg.norm(tip - target)), float(ang), False)


@dataclass
class CornerPlan:
    corner_id: int
    approach: IKResult
    contact: IKResult
    joint_jump_rad: float          # 접근 → 접촉 최대 관절 변화 (작아야 직선 하강)
    min_clear_m: float             # 손 링크(손끝 제외) 최저점 − 테이블
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def plan_session(model: HandModel, T_base_board: np.ndarray, table_z: float,
                 corner_ids: Sequence[int] | None = None, approach_m: float = 0.03,
                 tip_radius_m: float = 0.0, min_clear_m: float = 0.01,
                 max_jump_rad: float = 0.35) -> list[CornerPlan]:
    """코너마다 접근점 · 접촉점 IK. 손끝 링크 원점은 접촉점 + tip_radius 만큼 법선 위."""
    targets = plan_press_targets(T_base_board, corner_ids, approach_m)
    normal = targets[0].approach_base - targets[0].contact_base
    normal /= np.linalg.norm(normal)
    plans = []
    for t in targets:
        tip_c = t.contact_base + tip_radius_m * normal
        a = solve_press_ik(model, t.approach_base + tip_radius_m * normal, -normal)
        c = solve_press_ik(model, tip_c, -normal, seeds=(a.arm_q, *IK_SEEDS), q_ref=a.arm_q)
        hq = hand_q(c.arm_q)
        clear = min(model.link_pos(n, hq)[2] for n in CLEARANCE_LINKS) - table_z
        jump = float(np.max(np.abs(c.arm_q - a.arm_q)))
        p = CornerPlan(t.corner_id, a, c, jump, float(clear))
        if not a.ok:
            p.problems.append(f"접근점 IK 실패 (위치 {a.pos_err_m * 1e3:.1f} mm, 방향 {a.dir_err_deg:.1f}°)")
        if not c.ok:
            p.problems.append(f"접촉점 IK 실패 (위치 {c.pos_err_m * 1e3:.1f} mm, 방향 {c.dir_err_deg:.1f}°)")
        if jump > max_jump_rad:
            p.problems.append(f"접근→접촉 관절 변화 {jump:.2f} rad > {max_jump_rad}")
        if clear < min_clear_m:
            p.problems.append(f"손 링크가 테이블 여유 {clear * 1e3:.0f} mm < {min_clear_m * 1e3:.0f} mm")
        lim = [n for n, v in zip(ARM_JOINTS, c.arm_q)
               if min(v - model.limits[n][0], model.limits[n][1] - v) < 0.02]
        if lim:
            p.problems.append(f"관절 한계 0.02 rad 안: {lim}")
        plans.append(p)
    return plans


def substitute_corners(model: HandModel, T_base_board: np.ndarray, table_z: float,
                       want: int = 5, **kw) -> tuple[list[int], list[CornerPlan]]:
    """기본 5 코너 중 실패한 것을 닿는 코너로 바꾼다 (보드 중심에서 먼 순서로 넓게 퍼지게)."""
    plans = {p.corner_id: p for p in plan_session(model, T_base_board, table_z, **kw)}
    chosen = [i for i, p in plans.items() if p.ok]
    from . import corner_points_board
    P = corner_points_board()
    for i in sorted(range(N_CORNERS), key=lambda i: -np.linalg.norm(P[i, :2] - P[:, :2].mean(0))):
        if len(chosen) >= want:
            break
        if i in plans:
            continue
        plans[i] = plan_session(model, T_base_board, table_z, [i], **kw)[0]
        if plans[i].ok and all(np.linalg.norm(P[i] - P[j]) > 0.05 for j in chosen):
            chosen.append(i)
    return chosen, [plans[i] for i in chosen]


def _rot_small(rng: np.random.Generator, deg: float) -> np.ndarray:
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    return _rot_axis(v, np.radians(deg))


@dataclass
class LoopReport:
    board_pos_err_m: float
    board_rot_err_deg: float
    fit_rms_m: float
    injected_pos_m: float
    injected_rot_deg: float
    detected_pos_m: float
    detected_rot_deg: float
    corrected_pos_err_m: float
    corrected_rot_err_deg: float


def _pose_err(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    dR = A[:3, :3] @ B[:3, :3].T
    return (float(np.linalg.norm(A[:3, 3] - B[:3, 3])),
            float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))))


def recorded_tips(model: HandModel, plans: Sequence[CornerPlan], rng: np.random.Generator | None = None,
                  enc_sigma_rad: float = 0.0, contact_sigma_m: float = 0.0,
                  normal: np.ndarray | None = None) -> np.ndarray:
    """sim 접촉 관절각(+인코더 잡음) → FK 손끝. 접촉 위치 잡음은 보드 면 안(법선 수직)으로만."""
    out = []
    for p in plans:
        q = p.contact.arm_q + (rng.normal(0, enc_sigma_rad, 7) if rng is not None and enc_sigma_rad else 0)
        tip = model.link_pos(TIP_LINK, hand_q(q))
        if rng is not None and contact_sigma_m and normal is not None:
            e = rng.normal(0, contact_sigma_m, 3)
            tip = tip + e - (e @ normal) * normal
        out.append(tip)
    return np.array(out)


def closed_loop(model: HandModel, plans: Sequence[CornerPlan], T_base_board: np.ndarray,
                T_base_cam: np.ndarray, tip_radius_m: float = 0.0, inject_pos_m: float = 0.005,
                inject_rot_deg: float = 1.0, seed: int = 0) -> LoopReport:
    """잡음 없이: FK→적합이 정답과 맞는지, 틀린 extrinsics 를 compare_with_camera 가 찾고 고치는지."""
    rng = np.random.default_rng(seed)
    fit = fit_board_pose([p.corner_id for p in plans], recorded_tips(model, plans), tip_radius_m)
    pe, re = _pose_err(fit.T_base_board, T_base_board)
    T_cam_board = np.linalg.inv(T_base_cam) @ T_base_board            # 카메라가 정확히 본 보드
    bad = np.eye(4)
    bad[:3, :3] = _rot_small(rng, inject_rot_deg)
    d = rng.normal(size=3)
    bad[:3, 3] = inject_pos_m * d / np.linalg.norm(d)
    T_base_cam_bad = bad @ T_base_cam                                  # 틀어 놓은 extrinsics
    chk = compare_with_camera(T_base_cam_bad, T_cam_board, fit.T_base_board)
    cp, cr = _pose_err(chk.T_base_cam_corrected, T_base_cam)
    _, inj_rot = _pose_err(bad, np.eye(4))
    return LoopReport(pe, re, fit.rms_m, float(np.linalg.norm(_pose_err(T_base_cam_bad @ T_cam_board, T_base_board)[0])),
                      inj_rot, float(np.linalg.norm(chk.dpos_m)), chk.drot_deg, cp, cr)


@dataclass
class Sensitivity:
    enc_sigma_rad: float
    contact_sigma_m: float
    n_corners: int
    pos_p50_mm: float
    pos_p95_mm: float
    rot_p50_deg: float
    rot_p95_deg: float


def sensitivity(model: HandModel, plans: Sequence[CornerPlan], T_base_board: np.ndarray,
                enc_sigma_rad: float, contact_sigma_m: float, trials: int = 200,
                repeats: int = 1, tip_radius_m: float = 0.0, seed: int = 0) -> Sensitivity:
    """잡음 Monte-Carlo. repeats: 코너마다 몇 번 눌러 평균하는지."""
    rng = np.random.default_rng(seed)
    n = T_base_board[:3, 2] if T_base_board[2, 2] >= 0 else -T_base_board[:3, 2]
    ids = [p.corner_id for p in plans]
    pe, re = [], []
    for _ in range(trials):
        tips = np.mean([recorded_tips(model, plans, rng, enc_sigma_rad, contact_sigma_m, n)
                        for _ in range(repeats)], axis=0)
        a, b = _pose_err(fit_board_pose(ids, tips, tip_radius_m).T_base_board, T_base_board)
        pe.append(a)
        re.append(b)
    return Sensitivity(enc_sigma_rad, contact_sigma_m, len(ids),
                       float(np.percentile(pe, 50) * 1e3), float(np.percentile(pe, 95) * 1e3),
                       float(np.percentile(re, 50)), float(np.percentile(re, 95)))


def board_pose(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    """테이블 위 평평한 보드 (보드 z = base z)."""
    T = np.eye(4)
    T[:3, :3] = _rpy(0, 0, np.radians(yaw_deg))
    T[:3, 3] = [x, y, z]
    return T


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--board", nargs=4, type=float, metavar=("X", "Y", "Z", "YAW_DEG"), required=True,
                    help="보드 원점(코너 0 바깥 모서리) base=body_root 기준, 보드 면 높이 = Z")
    ap.add_argument("--table-z", type=float, help="테이블 상판 z (기본: 보드 Z)")
    ap.add_argument("--tip-radius", type=float, default=0.0)
    ap.add_argument("--corners", type=int, nargs="*")
    ap.add_argument("--urdf", default=str(DEFAULT_URDF))
    ap.add_argument("--json", help="보고서 저장 경로")
    a = ap.parse_args(argv)
    model = HandModel(a.urdf)
    T_board = board_pose(*a.board)
    table_z = a.board[2] if a.table_z is None else a.table_z
    kw = dict(tip_radius_m=a.tip_radius)
    if a.corners:
        plans = plan_session(model, T_board, table_z, a.corners, **kw)
        ids = [p.corner_id for p in plans if p.ok]
    else:
        ids, plans = substitute_corners(model, T_board, table_z, **kw)
    report: dict = {"corners": [{"id": p.corner_id, "ok": p.ok, "problems": p.problems,
                                 "contact_q": np.round(p.contact.arm_q, 4).tolist(),
                                 "approach_q": np.round(p.approach.arm_q, 4).tolist(),
                                 "clear_mm": round(p.min_clear_m * 1e3, 1)} for p in plans]}
    good = [p for p in plans if p.ok]
    for p in plans:
        print(f"corner {p.corner_id:2d}: {'OK ' if p.ok else 'NG '} 여유 {p.min_clear_m * 1e3:5.0f} mm  "
              f"점프 {p.joint_jump_rad:.2f}  {'; '.join(p.problems)}")
    if len(good) < 3:
        print("닿는 코너가 3개 미만 — 보드 자리를 옮기세요")
        report["ok"] = False
    else:
        cam = np.eye(4)
        cam[:3, 3] = [0.1, 0.0, 1.2]  # 닫힌 루프용 임의 카메라 (결과는 카메라 자세와 무관해야 한다)
        loop = closed_loop(model, good, T_board, cam, a.tip_radius)
        print(f"닫힌 루프: 보드 {loop.board_pos_err_m * 1e3:.3f} mm / {loop.board_rot_err_deg:.3f}°, "
              f"주입 {loop.injected_pos_m * 1e3:.1f} mm/{loop.injected_rot_deg:.2f}° → 검출 "
              f"{loop.detected_pos_m * 1e3:.1f} mm/{loop.detected_rot_deg:.2f}°, 보정 후 "
              f"{loop.corrected_pos_err_m * 1e3:.3f} mm/{loop.corrected_rot_err_deg:.3f}°")
        sens = [sensitivity(model, good, T_board, e, c, repeats=r)
                for e, c, r in ((0.002, 0.0, 1), (0.0, 0.002, 1), (0.002, 0.002, 1), (0.002, 0.002, 3))]
        for s, r in zip(sens, (1, 1, 1, 3)):
            print(f"잡음 enc {s.enc_sigma_rad:.3f} rad · 접촉 {s.contact_sigma_m * 1e3:.0f} mm · {r}회: "
                  f"위치 p50 {s.pos_p50_mm:.2f} / p95 {s.pos_p95_mm:.2f} mm, 회전 p50 {s.rot_p50_deg:.2f} / p95 {s.rot_p95_deg:.2f}°")
        report.update(ok=True, closed_loop=loop.__dict__, sensitivity=[s.__dict__ | {"repeats": r} for s, r in zip(sens, (1, 1, 1, 3))])
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
