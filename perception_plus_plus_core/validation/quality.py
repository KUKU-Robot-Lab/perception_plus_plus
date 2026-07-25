from dataclasses import dataclass
import numpy as np

from ..config import TrackingConfig
from ..types import FrameBundle, PoseResult
from .depth import valid_depth_mask
from .geometry import is_rigid_transform, pose_delta


@dataclass(frozen=True)
class QualityDecision:
    valid: bool
    reason: str = "OK"
    detail: str = ""


def evaluate_quality(
    frame: FrameBundle,
    result: PoseResult,
    previous: PoseResult | None,
    config: TrackingConfig,
) -> QualityDecision:
    pose = result.object_to_camera
    if not is_rigid_transform(pose):
        return QualityDecision(False, "NONFINITE_OR_INVALID_POSE")
    if result.mask is None or result.mask.shape != frame.depth.shape:
        return QualityDecision(False, "MASK_MISSING_OR_SHAPE")
    mask = result.mask.astype(bool)
    area = int(mask.sum())
    if area < config.min_mask_area_px:
        return QualityDecision(False, "MASK_TOO_SMALL", str(area))
    valid_depth = valid_depth_mask(frame.depth, config.min_depth_m, config.max_depth_m)
    ratio = float((valid_depth & mask).sum() / area)
    if ratio < config.min_valid_depth_ratio:
        return QualityDecision(False, "VALID_DEPTH_RATIO_LOW", f"{ratio:.3f}")
    position = pose[:3, 3]
    if np.any(position < config.workspace_min_xyz) or np.any(position > config.workspace_max_xyz):
        return QualityDecision(False, "OUTSIDE_WORKSPACE")
    if previous is not None:
        translation, rotation = pose_delta(previous.object_to_camera, pose)
        if translation > config.max_translation_m:
            return QualityDecision(False, "POSE_TRANSLATION_JUMP", f"{translation:.4f}")
        if rotation > config.max_rotation_deg:
            return QualityDecision(False, "POSE_ROTATION_JUMP", f"{rotation:.3f}")
    return QualityDecision(True)

