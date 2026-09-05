from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import numpy as np

from ..errors import DependencyUnavailable, ModelLoadError, classify_exception
from ..types import FrameBundle, MeshSpec, PoseResult


class FoundationPosePlusPlusAdapter:
    """Frame API around FP++.

    An engine can be injected for tests. Otherwise the upstream implementation
    is loaded lazily on first initialization, keeping CPU tools importable.
    """

    def __init__(self, upstream_root: str | Path = "external/foundationpose_plus_plus",
                 model_root: str | Path = "models", engine: Any | None = None,
                 estimate_iterations: int = 10,
                 track_iterations: int = 3, kalman_noise: float = 0.05) -> None:
        self.upstream_root = Path(upstream_root)
        self.model_root = Path(model_root)
        self.engine = engine
        self.estimate_iterations = estimate_iterations
        self.track_iterations = track_iterations
        self.kalman_noise = kalman_noise

    def _ensure_engine(self) -> Any:
        if self.engine is None:
            self.engine = _UpstreamEngine(
                self.upstream_root, self.model_root, self.estimate_iterations,
                self.track_iterations, self.kalman_noise)
        return self.engine

    @staticmethod
    def _result(value: Any, timestamp_ns: int) -> PoseResult:
        pose, mask = value
        if hasattr(pose, "detach"):
            pose = pose.detach().cpu().numpy()
        return PoseResult(np.asarray(pose).reshape(4, 4), np.asarray(mask).astype(bool),
                          timestamp_ns)

    def initialize(self, frame: FrameBundle, mask: np.ndarray, mesh: MeshSpec) -> PoseResult:
        try:
            value = self._ensure_engine().initialize(
                frame.rgb, frame.depth, mask, frame.intrinsics, mesh)
            return self._result(value, frame.timestamp_ns)
        except Exception as error:
            import traceback
            print("[fp_adapter] initialize failed:", repr(error), file=sys.stderr, flush=True)
            traceback.print_exc()
            raise classify_exception(error) from error

    def track(self, frame: FrameBundle) -> PoseResult:
        try:
            value = self._ensure_engine().track(frame.rgb, frame.depth, frame.intrinsics)
            return self._result(value, frame.timestamp_ns)
        except Exception as error:
            raise classify_exception(error) from error

    def reset(self) -> None:
        # ★엔진을 버리지 않는다(2026-09-03). 버리면 다음 initialize 가 FoundationPose +
        #   RasterizeCudaContext 를 다시 만들다 실패해 재등록마다 PERCEPTIONERROR 가 났다.
        #   추적 상태(칼만·마스크)만 비우고 모델·CUDA 컨텍스트는 재사용한다.
        if self.engine is not None:
            self.engine.reset()


class _UpstreamEngine:
    def __init__(self, root: Path, model_root: Path, estimate_iterations: int,
                 track_iterations: int, kalman_noise: float) -> None:
        if not root.is_dir():
            raise ModelLoadError(f"FP++ upstream directory is missing: {root}")
        self._link_models(root, model_root)
        for path in (root, root / "src", root / "FoundationPose", root / "Cutie"):
            if str(path.resolve()) not in sys.path:
                sys.path.insert(0, str(path.resolve()))
        try:
            import torch
            import trimesh
            from FoundationPose.estimater import (
                FoundationPose, PoseRefinePredictor, ScorePredictor, dr,
            )
            from VOT import Cutie
            from utils.kalman_filter_6d import KalmanFilter6D
            from obj_pose_track import (
                get_6d_pose_arr_from_mat, get_mat_from_6d_pose_arr,
                get_pose_xy_from_image_point,
            )
        except ImportError as error:
            raise DependencyUnavailable(f"FP++ dependency unavailable: {error}") from error
        self.torch, self.trimesh = torch, trimesh
        self.FoundationPose, self.Refiner = FoundationPose, PoseRefinePredictor
        self.Scorer, self.dr, self.Cutie = ScorePredictor, dr, Cutie
        self.Kalman = KalmanFilter6D
        self.get_pose_array = get_6d_pose_arr_from_mat
        self.get_pose_matrix = get_mat_from_6d_pose_arr
        self.get_pose_xy = get_pose_xy_from_image_point
        self.est_iter, self.track_iter, self.noise = (
            estimate_iterations, track_iterations, kalman_noise)
        self.estimator = self.cutie = self.kalman = None
        self.kf_mean = self.kf_covariance = None
        self.mask: np.ndarray | None = None

    @staticmethod
    def _link_models(root: Path, model_root: Path) -> None:
        mappings = {
            model_root / "foundationpose": root / "FoundationPose" / "weights",
            model_root / "cutie": root / "Cutie" / "cutie" / "weights",
        }
        for source_root, target_root in mappings.items():
            if not source_root.is_dir():
                continue
            for source in source_root.rglob("*"):
                if not source.is_file():
                    continue
                target = target_root / source.relative_to(source_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.symlink_to(source.resolve())

    def _load_mesh(self, mesh_spec: MeshSpec) -> Any:
        mesh = self.trimesh.load(str(mesh_spec.path))
        if isinstance(mesh, self.trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.apply_scale(mesh_spec.scale_to_meters)
        return mesh

    def initialize(self, rgb, depth, mask, intrinsics, mesh_spec):
        mesh = self._load_mesh(mesh_spec)
        if self.estimator is None:
            self.estimator = self.FoundationPose(
                model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                scorer=self.Scorer(), refiner=self.Refiner(),
                glctx=self.dr.RasterizeCudaContext())
            self.cutie = self.Cutie()
        else:
            # 재등록(LOST 복구): FoundationPose/RasterizeCudaContext 를 다시 만들면
            # 프로세스가 즉사한다(CUDA 이중 컨텍스트, 2026-09-02 재현). 기존 모델을
            # 재사용하고 메시·추적 상태만 리셋한다.
            self.estimator.reset_object(model_pts=mesh.vertices,
                                        model_normals=mesh.vertex_normals, mesh=mesh)
        self.kalman = self.Kalman(self.noise)
        self.mask = np.asarray(mask).astype(bool)
        pose = self.estimator.register(K=intrinsics.matrix, rgb=rgb, depth=depth,
                                       ob_mask=self.mask.astype(np.uint8) * 255,
                                       iteration=self.est_iter)
        self.cutie.initialize(rgb, {"mask": self.mask.astype(np.uint8)})
        self.kf_mean, self.kf_covariance = self.kalman.initiate(
            self.get_pose_array(pose))
        return pose, self.mask

    def track(self, rgb, depth, intrinsics):
        if self.estimator is None:
            raise RuntimeError("adapter is not initialized")
        from torchvision.transforms.functional import to_tensor
        with self.torch.no_grad():
            probability = self.cutie.cutie_processor.step(to_tensor(rgb).cuda().float())
            cutie_mask = self.cutie.cutie_processor.output_prob_to_mask(
                probability, segment_threshold=self.cutie.cutie_seg_threshold)
            self.mask = cutie_mask.detach().cpu().numpy().astype(bool)
        rows, cols = np.any(self.mask, axis=1), np.any(self.mask, axis=0)
        if np.any(rows) and np.any(cols):
            y0, y1 = np.where(rows)[0][[0, -1]]
            x0, x1 = np.where(cols)[0][[0, -1]]
            center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
            self.kf_mean, self.kf_covariance = self.kalman.update(
                self.kf_mean, self.kf_covariance,
                self.get_pose_array(self.estimator.pose_last))
            measurement_xy = np.asarray(self.get_pose_xy(
                self.estimator.pose_last, intrinsics.matrix, center_x, center_y))
            self.kf_mean, self.kf_covariance = self.kalman.update_from_xy(
                self.kf_mean, self.kf_covariance, measurement_xy)
            predicted = self.get_pose_matrix(self.kf_mean[:6])
            self.estimator.pose_last = self.torch.from_numpy(predicted).unsqueeze(0).to(
                self.estimator.pose_last.device)
        pose = self.estimator.track_one(rgb=rgb, depth=depth, K=intrinsics.matrix,
                                        iteration=self.track_iter)
        self.kf_mean, self.kf_covariance = self.kalman.predict(
            self.kf_mean, self.kf_covariance)
        return pose, self.mask

    def reset(self):
        # estimator·cutie 는 유지(재등록 시 reset_object + cutie.initialize 로 재사용).
        self.kalman = None
        self.kf_mean = self.kf_covariance = self.mask = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
