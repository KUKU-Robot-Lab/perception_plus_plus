import numpy as np


def depth_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
    if encoding in {"16UC1", "mono16"}:
        return depth.astype(np.float32) * 0.001
    if encoding == "32FC1":
        return depth.astype(np.float32, copy=False)
    raise ValueError(f"unsupported depth encoding: {encoding}")


def valid_depth_mask(depth: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)

