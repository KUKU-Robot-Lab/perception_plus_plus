import numpy as np


def is_rigid_transform(pose: np.ndarray, atol: float = 1e-3) -> bool:
    if np.shape(pose) != (4, 4) or not np.isfinite(pose).all():
        return False
    rotation = pose[:3, :3]
    return (
        np.allclose(rotation.T @ rotation, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=atol)
        and np.allclose(pose[3], [0, 0, 0, 1], atol=atol)
    )


def pose_delta(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
    relative = previous[:3, :3].T @ current[:3, :3]
    cosine = np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)
    return translation, float(np.degrees(np.arccos(cosine)))

