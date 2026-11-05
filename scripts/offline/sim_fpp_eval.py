#!/usr/bin/env python3
"""sim_fpp_batch 결과(ep_XXXX_fpp.npz)를 렌더 GT(ep_XXXX.npz)와 비교해 오차를 요약한다.

프레임별: 위치 오차 mm(전체·대칭축 방향·축 직각), 회전 오차 deg(전체, 대칭 고려 = 대칭축 기울기).
축대칭 물체(컵)는 축 둘레 회전이 관측 불가이므로 '대칭 고려 회전 오차' = GT 축과 추정 축 사이 각.
대칭축은 물체 프레임 기준으로 --sym-axis, 없으면 meta.symmetry_axis, 없으면 0,0,1.
메시 프레임 축이면 meta.T_obj_mesh 로 물체 프레임으로 돌린다(--axis-frame mesh).

집계: 전체·상태별·visible_frac 구간별 오차 중앙값/p90 과 LOST 비율, 첫 TRACKING 까지 프레임 수.
GPU·ROS 없이 numpy 만으로 돈다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

VIS_BINS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0001)


def _meta(raw) -> dict:
    if raw is None:
        return {}
    v = raw.item() if isinstance(raw, np.ndarray) else raw
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    try:
        return json.loads(v) if isinstance(v, str) else dict(v)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"_raw": str(v)}


def sym_axis_obj(args, gt_meta: dict) -> np.ndarray:
    if args.sym_axis:
        axis = np.array([float(x) for x in args.sym_axis.split(",")])
    else:
        axis = np.asarray(gt_meta.get("symmetry_axis", [0.0, 0.0, 1.0]), float)
    if axis.shape != (3,) or np.linalg.norm(axis) < 1e-9:
        raise ValueError(f"bad symmetry axis {axis}")
    axis = axis / np.linalg.norm(axis)
    if args.axis_frame == "mesh" and gt_meta.get("T_obj_mesh") is not None:
        axis = np.asarray(gt_meta["T_obj_mesh"], float)[:3, :3] @ axis
    return axis


def frame_errors(T_gt: np.ndarray, T_fp: np.ndarray, axis: np.ndarray) -> dict:
    n = len(T_gt)
    ok = np.isfinite(T_gt).all(axis=(1, 2)) & np.isfinite(T_fp).all(axis=(1, 2))
    out = {k: np.full(n, np.nan) for k in
           ("pos_mm", "pos_axial_mm", "pos_radial_mm", "rot_deg", "rot_sym_deg", "depth_mm")}
    if not ok.any():
        return out
    g, f = T_gt[ok], T_fp[ok]
    d = f[:, :3, 3] - g[:, :3, 3]
    a_gt = np.einsum("tij,j->ti", g[:, :3, :3], axis)          # 카메라 프레임 대칭축(GT)
    a_fp = np.einsum("tij,j->ti", f[:, :3, :3], axis)
    axial = np.einsum("ti,ti->t", d, a_gt)
    R_rel = np.einsum("tji,tjk->tik", g[:, :3, :3], f[:, :3, :3])
    tr = np.clip((np.trace(R_rel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    cos_ax = np.clip(np.einsum("ti,ti->t", a_gt, a_fp), -1, 1)
    out["pos_mm"][ok] = np.linalg.norm(d, axis=1) * 1e3
    out["pos_axial_mm"][ok] = axial * 1e3
    out["pos_radial_mm"][ok] = np.linalg.norm(d - axial[:, None] * a_gt, axis=1) * 1e3
    out["rot_deg"][ok] = np.degrees(np.arccos(tr))
    out["rot_sym_deg"][ok] = np.degrees(np.arccos(cos_ax))
    out["depth_mm"][ok] = d[:, 2] * 1e3                          # 카메라 z(깊이) 방향 부호 있는 오차
    return out


def _stat(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "mean": float(v.mean()), "max": float(v.max())}


def summarize(err: dict, status: np.ndarray, valid: np.ndarray, vis: np.ndarray,
              gt_present: np.ndarray) -> dict:
    keys = ("pos_mm", "pos_radial_mm", "rot_sym_deg", "rot_deg")
    abs_axial = np.abs(err["pos_axial_mm"])
    res = {"frames": int(len(status)),
           "status_counts": {s: int((status == s).sum()) for s in np.unique(status)},
           "valid_frac": float(valid.mean()) if len(valid) else 0.0}
    sel = valid & gt_present
    res["valid_err"] = {k: _stat(err[k][sel]) for k in keys}
    res["valid_err"]["abs_axial_mm"] = _stat(abs_axial[sel])
    res["valid_err"]["depth_mm_signed"] = _stat(err["depth_mm"][sel])
    first = np.flatnonzero(status == "TRACKING")
    res["first_tracking_frame"] = int(first[0]) if first.size else None
    bins = []
    for lo, hi in zip(VIS_BINS[:-1], VIS_BINS[1:]):
        m = gt_present & (vis >= lo) & (vis < hi)
        if not m.any():
            continue
        bins.append({"vis": f"[{lo:.2f},{min(hi, 1.0):.2f}]", "frames": int(m.sum()),
                     "lost_frac": float((status[m] == "LOST").mean()),
                     "valid_frac": float(valid[m].mean()),
                     "pos_mm": _stat(err["pos_mm"][m & valid]),
                     "rot_sym_deg": _stat(err["rot_sym_deg"][m & valid])})
    res["by_visible_frac"] = bins
    absent = ~gt_present
    if absent.any():  # 물체가 화면에 없는데 valid 자세를 낸 프레임 = 거짓 양성
        res["absent_frames"] = int(absent.sum())
        res["absent_but_valid"] = int((absent & valid).sum())
    return res


def eval_pair(gt_path: Path, fp_path: Path, args) -> tuple[dict, list[dict]]:
    with np.load(gt_path, allow_pickle=False) as g, np.load(fp_path, allow_pickle=False) as f:
        gt_meta = _meta(g["meta"]) if "meta" in g.files else {}
        fp_meta = _meta(f["meta"]) if "meta" in f.files else {}
        n = len(f["status"])
        start = int(f["frame_index"][0]) if "frame_index" in f.files and n else 0
        sl = slice(start, start + n)
        T_gt = np.asarray(g["T_cam_obj_gt"], float)[sl]
        vis = (np.asarray(g["visible_frac"], float)[sl] if "visible_frac" in g.files
               else np.ones(n))
        mask_area_gt = (np.asarray(g["mask_gt"])[sl].reshape(n, -1) > 0).sum(1) \
            if "mask_gt" in g.files else np.full(n, -1)
        T_fp = np.asarray(f["T_cam_obj_fp"], float)
        status = np.asarray(f["status"]).astype(str)
        reason = np.asarray(f["reason"]).astype(str)
        valid = np.asarray(f["valid"], bool)
        ms = np.asarray(f["ms"], float)
    gt_present = np.isfinite(T_gt).all(axis=(1, 2)) & (vis > 0)
    axis = sym_axis_obj(args, gt_meta)
    err = frame_errors(T_gt, T_fp, axis)
    summ = summarize(err, status, valid, vis, gt_present)
    summ.update({"episode": gt_path.name, "fpp": fp_path.name, "sym_axis_obj": axis.tolist(),
                 "mask_mode": fp_meta.get("mask_mode"),
                 "ms": _stat(ms), "reasons": {r: int((reason == r).sum()) for r in np.unique(reason)}})
    rows = []
    for i in range(n):
        rows.append({"episode": gt_path.stem, "frame": start + i, "status": status[i],
                     "reason": reason[i], "valid": int(valid[i]), "visible_frac": round(vis[i], 4),
                     "mask_area_gt": int(mask_area_gt[i]), "ms": round(ms[i], 2),
                     **{k: (round(float(v[i]), 3) if np.isfinite(v[i]) else "") for k, v in err.items()}})
    return summ, rows


def pair_inputs(gt_dir: Path, fp_dir: Path, tag: str) -> list[tuple[Path, Path]]:
    pairs = []
    for fp in sorted(fp_dir.glob(f"ep_*{tag}.npz")):
        stem = fp.name[: -len(f"{tag}.npz")]
        gt = gt_dir / f"{stem}.npz"
        if gt.exists():
            pairs.append((gt, fp))
        else:
            print(f"[eval] skip {fp.name}: no GT {gt}", file=sys.stderr)
    return pairs


def _fmt(s: dict, key: str) -> str:
    v = s.get(key, {})
    return f"{v['median']:.1f}/{v['p90']:.1f}" if v.get("n") else "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gt-dir", required=True, help="ep_XXXX.npz 렌더 디렉터리")
    ap.add_argument("--fpp-dir", required=True, help="ep_XXXX_fpp.npz 디렉터리")
    ap.add_argument("--tag", default="_fpp", help="결과 파일 접미사(ep_0000<tag>.npz)")
    ap.add_argument("--sym-axis", default="", help="물체 프레임 대칭축 x,y,z(기본 meta 또는 0,0,1)")
    ap.add_argument("--axis-frame", choices=("obj", "mesh"), default="mesh",
                    help="meta.symmetry_axis 가 메시 프레임이면 mesh(T_obj_mesh 회전 적용)")
    ap.add_argument("--out", default="", help="요약 json 경로(기본 <fpp-dir>/eval<tag>.json)")
    args = ap.parse_args(argv)

    gt_dir, fp_dir = Path(args.gt_dir), Path(args.fpp_dir)
    pairs = pair_inputs(gt_dir, fp_dir, args.tag)
    if not pairs:
        print("[eval] no (GT, fpp) pairs", file=sys.stderr)
        return 1
    summaries, rows = [], []
    for gt, fp in pairs:
        try:
            s, r = eval_pair(gt, fp, args)
        except (KeyError, ValueError, OSError) as exc:
            print(f"[eval] {fp.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        summaries.append(s)
        rows.extend(r)
        print(f"[eval] {gt.stem}: frames {s['frames']} valid {s['valid_frac']:.2f} "
              f"status {s['status_counts']} | pos mm med/p90 {_fmt(s['valid_err'], 'pos_mm')} "
              f"axial {_fmt(s['valid_err'], 'abs_axial_mm')} radial {_fmt(s['valid_err'], 'pos_radial_mm')} "
              f"| rot_sym deg {_fmt(s['valid_err'], 'rot_sym_deg')} | ms {_fmt(s, 'ms')}")
        for b in s["by_visible_frac"]:
            print(f"        vis {b['vis']}: n {b['frames']} LOST {b['lost_frac']:.2f} "
                  f"valid {b['valid_frac']:.2f} pos {_fmt(b, 'pos_mm')} rot {_fmt(b, 'rot_sym_deg')}")
    if not summaries:
        return 1
    out = Path(args.out) if args.out else fp_dir / f"eval{args.tag}.json"
    out.write_text(json.dumps({"episodes": summaries}, ensure_ascii=False, indent=1))
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[eval] wrote {out} and {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
