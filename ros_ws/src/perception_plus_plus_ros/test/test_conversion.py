import numpy as np

from perception_plus_plus_ros.conversion import rotation_matrix_to_quaternion


def test_identity_rotation_is_identity_quaternion():
    assert rotation_matrix_to_quaternion(np.eye(3)) == (0.0, 0.0, 0.0, 1.0)

