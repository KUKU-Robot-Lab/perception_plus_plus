#!/usr/bin/env python3
"""sim 에서 렌더한 RGB-D 에피소드(ep_XXXX.npz)에 진짜 FP++ 를 오프라인으로 돌린다.

라이브 cup_tracking 노드(ros_ws/.../node.py CupTrackingNode)와 같은 객체를 ROS 없이 엮는다:
  TrackingManager(FoundationPosePlusPlusAdapter(), detector, MeshSpec, TrackingConfig.from_yaml)
  -> 프레임마다 manager.process(FrameBundle) 한 번.
유효성 검사(evaluate_quality), LOST 전이, recovery_interval·reinitialize_valid_frames 재초기화는
매니저 코드 그대로다. 달라지는 것은 첫·재초기화 마스크의 출처(detector)뿐이다:
  --mask-mode yolo : 라이브와 같은 YoloCupDetector(weights, class_id, conf, pick)
  --mask-mode gt   : 그 프레임의 sim 분할 마스크(mask_gt > 0)를 검출 하나로 돌려준다

입력 계약(렌더 쪽): rgb (T,H,W,3) uint8 · depth (T,H,W) float32 m(무효 0) · K (3,3) ·
mask_gt (T,H,W) uint8 · T_cam_obj_gt (T,4,4) optical · tick · t_s · visible_frac · meta(json).
meta 의 T_obj_mesh(4x4, 선택)는 물체 원점 -> FP++ 메시 원점 변환이다. FP++ 는 메시 프레임
자세(T_cam_mesh)를 내므로 출력 T_cam_obj_fp = T_cam_mesh_fp @ inv(T_obj_mesh).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

REPO = Path(os.environ.get("PPP_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO))

from perception_plus_plus_core.config import TrackingConfig  # noqa: E402
from perception_plus_plus_core.detection.base import Detection  # noqa: E402
from perception_plus_plus_core.tracking.manager import TrackingManager  # noqa: E402
from perception_plus_plus_core.tracking.state import TrackingState  # noqa: E402
from perception_plus_plus_core.types import CameraIntrinsics, FrameBundle, MeshSpec  # noqa: E402

REQUIRED_KEYS = ("rgb", "depth", "K", "T_cam_obj_gt", "t_s")
# 라이브 node.py 의 파라미터 기본값(파라미터 파일이 없을 때)
LIVE_DEFAULTS = {
    "mesh_path": "assets/meshes/cup.obj",
    "mesh_scale_to_meters": 1.0,
    "yolo_weights": "models/yolo/yolo11n.pt",
    "cup_class_id": 41,
    "yolo_confidence": 0.25,
    "detection_pick": "confidence",
    "tracking_config": "config/cup_tracking.yaml",
}
STATUS_LEN = 16
REASON_LEN = 40


class GtMaskDetector:
    """sim 분할 마스크를 라이브 검출기 자리에 꽂는다. 마스크가 비면 검출 없음."""

    def __init__(self) -> None:
        self.mask: np.ndarray | None = None
        self.calls = 0

    def set_frame(self, mask: np.ndarray | None) -> None:
        self.mask = mask

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        self.calls += 1
        if self.mask is None:
            return []
        m = self.mask > 0
        if m.shape != rgb.shape[:2] or not m.any():
            return []
        ys, xs = np.nonzero(m)
        box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        return [Detection(m, 1.0, -1, box)]


class RecordingDetector:
    """검출 호출과 첫 검출의 마스크 면적을 기록한다(동작은 그대로)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.last: list[Detection] | None = None

    def set_frame(self, mask) -> None:
        if hasattr(self.inner, "set_frame"):
            self.inner.set_frame(mask)

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        self.last = self.inner.detect(rgb)
        return self.last


class RecordingAdapter:
    """어댑터 호출(init/track)과 검증 전 원시 결과를 기록한다(동작은 그대로)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.call = ""
        self.raw = None
        self.resets = 0

    def initialize(self, frame, mask, mesh):
        self.call = "init"
        self.raw = None
        self.raw = self.inner.initialize(frame, mask, mesh)
        return self.raw

    def track(self, frame):
        self.call = "track"
        self.raw = None
        self.raw = self.inner.track(frame)
        return self.raw

    def reset(self) -> None:
        self.resets += 1
        self.inner.reset()


def _md5(path: Path, limit: int | None = None) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read() if limit is None else f.read(limit))
    return h.hexdigest()


def load_params(path: str | None) -> dict:
    params = dict(LIVE_DEFAULTS)
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if "cup_tracking" in raw:
            raw = raw["cup_tracking"].get("ros__parameters", {})
        unknown = set(raw) - set(LIVE_DEFAULTS) - {
            "rgb_topic", "depth_topic", "camera_info_topic", "pose_topic", "status_topic",
            "child_frame_id", "sync_slop_seconds", "sync_queue_size"}
        if unknown:
            raise ValueError(f"unknown params keys in {path}: {sorted(unknown)}")
        params.update({k: v for k, v in raw.items() if k in LIVE_DEFAULTS})
    return params


def parse_meta(raw) -> dict:
    if raw is None:
        return {}
    value = raw.item() if isinstance(raw, np.ndarray) else raw
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, dict):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return {"_raw": str(value)}


def load_episode(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"{path.name}: missing keys {missing}")
        ep = {k: z[k] for k in z.files if k != "meta"}
        ep["meta"] = parse_meta(z["meta"]) if "meta" in z.files else {}
    t = ep["rgb"].shape[0]
    if ep["rgb"].ndim != 4 or ep["rgb"].shape[-1] != 3 or ep["rgb"].dtype != np.uint8:
        raise ValueError(f"{path.name}: rgb must be uint8 (T,H,W,3), got {ep['rgb'].dtype} {ep['rgb'].shape}")
    if ep["depth"].shape != ep["rgb"].shape[:3]:
        raise ValueError(f"{path.name}: depth shape {ep['depth'].shape} != rgb {ep['rgb'].shape[:3]}")
    if "mask_gt" in ep and ep["mask_gt"].shape != ep["rgb"].shape[:3]:
        raise ValueError(f"{path.name}: mask_gt shape {ep['mask_gt'].shape} mismatch")
    if ep["T_cam_obj_gt"].shape != (t, 4, 4):
        raise ValueError(f"{path.name}: T_cam_obj_gt shape {ep['T_cam_obj_gt'].shape}")
    if np.asarray(ep["K"]).shape != (3, 3):
        raise ValueError(f"{path.name}: K must be (3,3)")
    return ep


def resolve_mesh(params: dict, meta: dict, args) -> tuple[str, float]:
    """--mesh > 파라미터 파일 > meta(fpp_mesh) 순."""
    if args.mesh:
        return args.mesh, float(args.mesh_scale if args.mesh_scale else params["mesh_scale_to_meters"])
    if args.params:
        return params["mesh_path"], float(params["mesh_scale_to_meters"])
    name = meta.get("fpp_mesh") or meta.get("mesh")
    if name:
        base = Path(name).name  # 렌더 쪽 meta 는 "sim2real/assets/meshes/x.obj" 처럼 저장소 상대 경로를 준다
        cands = [Path(name), REPO / "assets/s2r_meshes" / name, REPO / "assets/meshes" / name,
                 REPO / "assets/s2r_meshes" / base, REPO / "assets/meshes" / base,
                 REPO / "assets/s2r_meshes" / f"{name}.obj", REPO / "assets/meshes" / f"{name}.obj"]
        for c in cands:
            if (c if c.is_absolute() else REPO / c).is_file():
                scale = meta.get("fpp_mesh_scale", meta.get("mesh_scale_to_meters", 1.0))
                return str(c), float(scale)
        raise FileNotFoundError(f"meta mesh {name!r} not found under assets/(s2r_)meshes")
    return params["mesh_path"], float(params["mesh_scale_to_meters"])


def pick_params_file(params_dir: str, meta: dict) -> str | None:
    """에피소드마다 라이브 파라미터 파일을 고른다.
    1) meta.object_real 첫 낱말(예: source240b_blue) == <params_dir>/<이름>.yaml
    2) meta.fpp_mesh 의 파일 이름과 mesh_path 파일 이름이 같은 첫 파일(이름순)."""
    d = Path(params_dir)
    real = str(meta.get("object_real") or "").split("(")[0].split()[0:1]
    if real and (d / f"{real[0]}.yaml").is_file():
        return str(d / f"{real[0]}.yaml")
    mesh = Path(str(meta.get("fpp_mesh") or meta.get("mesh") or "")).name
    if not mesh:
        return None
    for f in sorted(d.glob("*.yaml")):
        raw = yaml.safe_load(f.read_text()) or {}
        p = raw.get("cup_tracking", {}).get("ros__parameters", raw)
        if Path(str(p.get("mesh_path", ""))).name == mesh:
            return str(f)
    return None


def build_detector(mode: str, params: dict):
    if mode == "gt":
        return GtMaskDetector()
    from perception_plus_plus_core.detection.yolo import YoloCupDetector
    return YoloCupDetector(params["yolo_weights"], int(params["cup_class_id"]),
                           float(params["yolo_confidence"]), pick=params["detection_pick"])


def _t_obj_mesh(meta: dict) -> np.ndarray:
    value = meta.get("T_obj_mesh")
    if value is None:
        return np.eye(4)
    m = np.asarray(value, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"meta T_obj_mesh must be 4x4, got {m.shape}")
    return m


def run_episode(ep: dict, adapter, detector, mesh: MeshSpec, config: TrackingConfig,
                args) -> dict:
    t_total = ep["rgb"].shape[0]
    start = args.start
    stop = t_total if args.max_frames <= 0 else min(t_total, start + args.max_frames)
    n = stop - start
    K = np.asarray(ep["K"], dtype=np.float64)
    h, w = ep["rgb"].shape[1:3]
    intr = CameraIntrinsics(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), w, h)
    inv_obj_mesh = np.linalg.inv(_t_obj_mesh(ep["meta"]))

    rec_adapter = RecordingAdapter(adapter)
    rec_detector = RecordingDetector(detector)
    adapter.reset()  # 에피소드 사이 추적 상태 비움(엔진·모델은 유지 — 라이브 LOST 재초기화와 같은 경로)
    manager = TrackingManager(rec_adapter, rec_detector, mesh, config)

    out = {
        "T_cam_obj_fp": np.full((n, 4, 4), np.nan),
        "T_cam_mesh_raw": np.full((n, 4, 4), np.nan),
        "status": np.empty(n, dtype=f"<U{STATUS_LEN}"),
        "reason": np.empty(n, dtype=f"<U{REASON_LEN}"),
        "valid": np.zeros(n, dtype=bool),
        "mask_area": np.zeros(n, dtype=np.int32),
        "det_area": np.full(n, -1, dtype=np.int32),
        "call": np.empty(n, dtype="<U5"),
        "consecutive_valid": np.zeros(n, dtype=np.int32),
        "consecutive_invalid": np.zeros(n, dtype=np.int32),
        "fatal": np.zeros(n, dtype=bool),
        "ms": np.zeros(n, dtype=np.float64),
        "frame_index": np.arange(start, stop, dtype=np.int32),
    }
    for key in ("tick", "t_s", "visible_frac"):
        if key in ep:
            out[key] = np.asarray(ep[key])[start:stop]
    mask_gt = ep.get("mask_gt")
    t_s = np.asarray(ep["t_s"], dtype=np.float64)
    for i, f in enumerate(range(start, stop)):
        frame = FrameBundle(ep["rgb"][f], np.asarray(ep["depth"][f], dtype=np.float32), intr,
                            int(round(t_s[f] * 1e9)), "camera_color_optical_frame")
        rec_detector.set_frame(None if mask_gt is None else mask_gt[f])
        rec_detector.last, rec_adapter.call, rec_adapter.raw = None, "", None
        if args.sync_cuda:
            _cuda_sync()
        t0 = time.perf_counter()
        output = manager.process(frame)
        if args.sync_cuda:
            _cuda_sync()
        out["ms"][i] = (time.perf_counter() - t0) * 1e3
        out["status"][i] = output.state.name
        out["reason"][i] = output.reason[:REASON_LEN]
        out["consecutive_valid"][i] = output.consecutive_valid
        out["consecutive_invalid"][i] = output.consecutive_invalid
        out["fatal"][i] = output.fatal
        out["call"][i] = rec_adapter.call
        if rec_detector.last:
            out["det_area"][i] = int(np.count_nonzero(rec_detector.last[0].mask))
        elif rec_detector.last is not None:
            out["det_area"][i] = 0
        raw = rec_adapter.raw
        if raw is not None:
            out["T_cam_mesh_raw"][i] = raw.object_to_camera
            if raw.mask is not None:
                out["mask_area"][i] = int(np.count_nonzero(raw.mask))
        if output.pose is not None:  # 라이브 노드가 pose 를 publish 하는 조건과 같다
            out["valid"][i] = True
            out["T_cam_obj_fp"][i] = np.asarray(output.pose.object_to_camera) @ inv_obj_mesh
    out["adapter_resets"] = np.int32(rec_adapter.resets)
    return out


def _cuda_sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - 시간 측정 보조일 뿐
        pass


def _summary(out: dict) -> dict:
    status, ms = out["status"], out["ms"]
    names = [s.name for s in TrackingState]
    reasons, counts = np.unique(out["reason"], return_counts=True)
    calls = out["call"]
    def stat(v):
        return {"n": int(v.size), "median": float(np.median(v)) if v.size else None,
                "p90": float(np.percentile(v, 90)) if v.size else None,
                "mean": float(v.mean()) if v.size else None}
    return {
        "frames": int(status.size),
        "status_frac": {s: float(np.mean(status == s)) for s in names},
        "valid_frac": float(out["valid"].mean()) if status.size else 0.0,
        "reasons": {str(r): int(c) for r, c in zip(reasons, counts)},
        "ms_track": stat(ms[calls == "track"]),
        "ms_init": stat(ms[calls == "init"]),
        "ms_all": stat(ms),
        "adapter_resets": int(out["adapter_resets"]),
        "fatal": bool(out["fatal"].any()),
    }


def gather_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            # 계약 이름 ep_XXXX.npz 만(렌더 쪽 ep_XXXX_trace.npz·결과 *_fpp.npz 는 제외)
            paths += sorted(x for x in p.glob("ep_*.npz") if re.fullmatch(r"ep_\d+\.npz", x.name))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(item)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help="ep_XXXX.npz 또는 그 디렉터리")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mask-mode", choices=("gt", "yolo"), default="gt")
    ap.add_argument("--params", help="라이브 노드 파라미터 yaml(sim2real object_registry 가 렌더한 것)")
    ap.add_argument("--params-dir", help="에피소드마다 meta(object_real·fpp_mesh)로 고를 파라미터 yaml 디렉터리"
                    "(--params 보다 우선; 못 찾으면 --params/기본값)")
    ap.add_argument("--tracking-config", help="기본: 파라미터의 tracking_config(라이브와 같은 config/cup_tracking.yaml)")
    ap.add_argument("--mesh", help="메시 경로 강제(기본: --params > meta.fpp_mesh)")
    ap.add_argument("--mesh-scale", type=float)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = 전부")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--sync-cuda", action="store_true", default=True,
                    help="프레임 시간 측정 전후 cuda synchronize(기본 켬)")
    ap.add_argument("--tag", default="", help="출력 파일 이름 접미사(예: _gt)")
    args = ap.parse_args(argv)

    paths = gather_inputs(args.inputs)
    if not paths:
        print("입력 에피소드 없음", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = load_params(args.params)
    cfg_path = args.tracking_config or params["tracking_config"]
    config = TrackingConfig.from_yaml(cfg_path)

    from perception_plus_plus_core.fp_adapter.foundationpose_plus_plus import (
        FoundationPosePlusPlusAdapter)
    adapter = FoundationPosePlusPlusAdapter()  # 라이브 node.py 와 같은 기본값
    detectors: dict = {}

    def detector_for(p: dict):
        key = (args.mask_mode, p["yolo_weights"], p["cup_class_id"], p["yolo_confidence"], p["detection_pick"])
        if key not in detectors:
            detectors[key] = build_detector(args.mask_mode, p)
        return detectors[key]
    t0 = time.perf_counter()
    if hasattr(adapter, "_ensure_engine"):
        adapter._ensure_engine()  # 엔진 적재 시간을 프레임 시간에서 뺀다
    engine_load_s = time.perf_counter() - t0
    print(f"[simfpp] engine load {engine_load_s:.1f}s · mode={args.mask_mode} · config={cfg_path}", flush=True)

    run_meta_common = {
        "image": os.environ.get("SIMFPP_IMAGE", ""),
        "ppp_commit": os.environ.get("SIMFPP_PPP_COMMIT", ""),
        "ppp_live_mounts_md5": os.environ.get("SIMFPP_LIVE_MD5", ""),
        "host": os.environ.get("SIMFPP_HOST", ""),
        "mask_mode": args.mask_mode,
        "tracking_config_path": cfg_path,
        "tracking_config": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in vars(config).items()},
        "params_file": args.params or "",
        "params": params,
        "engine_load_s": engine_load_s,
        "adapter": {k: getattr(adapter, k, None) for k in
                    ("estimate_iterations", "track_iterations", "kalman_noise")},
        "rules": "TrackingManager (perception_plus_plus_core.tracking.manager) = live CupTrackingNode",
    }
    failures = 0
    for path in paths:
        dst = out_dir / f"{path.stem}{args.tag}_fpp.npz"
        if args.skip_existing and dst.exists():
            print(f"[simfpp] skip {dst.name}", flush=True)
            continue
        try:
            ep = load_episode(path)
            ep_params_file = args.params or ""
            ep_params = params
            if args.params_dir:
                f = pick_params_file(args.params_dir, ep["meta"])
                if f:
                    ep_params_file, ep_params = f, load_params(f)
            ep_args = argparse.Namespace(**{**vars(args), "params": ep_params_file or None})
            mesh_path, scale = resolve_mesh(ep_params, ep["meta"], ep_args)
            detector = detector_for(ep_params)
            mesh = MeshSpec(mesh_path, scale)
            if args.mask_mode == "gt" and "mask_gt" not in ep:
                raise ValueError(f"{path.name}: --mask-mode gt needs mask_gt")
            t_ep = time.perf_counter()
            out = run_episode(ep, adapter, detector, mesh, config, args)
            summary = _summary(out)
            meta = dict(run_meta_common, input=str(path), input_md5_head=_md5(path, 1 << 24),
                        mesh_path=mesh_path, mesh_scale=scale,
                        params_file=ep_params_file, params=ep_params,
                        T_obj_mesh=_t_obj_mesh(ep["meta"]).tolist(),
                        episode_meta=ep["meta"], summary=summary,
                        wall_s=time.perf_counter() - t_ep,
                        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            tmp = dst.with_name(dst.name + ".tmp.npz")
            np.savez_compressed(tmp, meta=json.dumps(meta, ensure_ascii=False, default=str), **out)
            os.replace(tmp, dst)
            print(f"[simfpp] {path.name}: {summary['frames']} fr · TRACKING "
                  f"{summary['status_frac']['TRACKING']:.2f} · valid {summary['valid_frac']:.2f} · "
                  f"track ms med {summary['ms_track']['median']} · init ms med "
                  f"{summary['ms_init']['median']} -> {dst}", flush=True)
            if summary["fatal"]:
                print(f"[simfpp] FATAL in {path.name}: {summary['reasons']}", file=sys.stderr)
                failures += 1
                break  # 라이브와 같이 치명 오류 뒤에는 매니저가 멈춘다 — 다음 에피소드도 의미 없음
        except Exception:  # noqa: BLE001 - 에피소드 하나 실패가 배치를 멈추지 않게
            failures += 1
            traceback.print_exc()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
