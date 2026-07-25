import numpy as np


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    m = matrix
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    index = int(np.argmax(np.diag(m)))
    if index == 0:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    if index == 1:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s,
                (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
            0.25 * s, (m[1, 0] - m[0, 1]) / s)


def fill_transform(message, matrix: np.ndarray) -> None:
    translation = message.position if hasattr(message, "position") else message.translation
    rotation = message.orientation if hasattr(message, "orientation") else message.rotation
    translation.x, translation.y, translation.z = (
        float(v) for v in matrix[:3, 3])
    quaternion = rotation_matrix_to_quaternion(matrix[:3, :3])
    (rotation.x, rotation.y, rotation.z, rotation.w) = (
        float(v) for v in quaternion)
