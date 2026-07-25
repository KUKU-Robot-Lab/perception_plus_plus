from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class TrackingConfig:
    min_depth_m: float = 0.1
    max_depth_m: float = 3.0
    min_mask_area_px: int = 20
    min_valid_depth_ratio: float = 0.5
    max_translation_m: float = 0.25
    max_rotation_deg: float = 75.0
    workspace_min_xyz: tuple[float, float, float] = (-2.0, -2.0, 0.05)
    workspace_max_xyz: tuple[float, float, float] = (2.0, 2.0, 3.0)
    max_invalid_frames: int = 3
    recovery_interval_frames: int = 15
    reinitialize_valid_frames: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.min_depth_m < self.max_depth_m:
            raise ValueError("invalid depth range")
        if not 0 <= self.min_valid_depth_ratio <= 1:
            raise ValueError("valid depth ratio must be within [0, 1]")
        if min(self.max_invalid_frames, self.recovery_interval_frames,
               self.reinitialize_valid_frames) < 1:
            raise ValueError("frame counters must be positive")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "TrackingConfig":
        known = {field.name for field in fields(cls)}
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"unknown tracking config keys: {sorted(unknown)}")
        values = dict(mapping)
        for key in ("workspace_min_xyz", "workspace_max_xyz"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrackingConfig":
        return cls.from_mapping(yaml.safe_load(Path(path).read_text()) or {})

