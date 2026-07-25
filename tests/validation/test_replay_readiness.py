import json
import numpy as np

from perception_plus_plus_core.validation.readiness import CheckResult, readiness_status
from perception_plus_plus_core.validation.replay import ReplayReader


def test_replay_reader_is_ordered(tmp_path):
    np.savez_compressed(tmp_path / "frames.npz", rgb=np.zeros((2, 2, 2, 3), np.uint8),
                        depth=np.ones((2, 2, 2), np.float32),
                        timestamps_ns=np.array([20, 10]), K=np.eye(3))
    reader = ReplayReader(tmp_path / "frames.npz", "camera")
    assert [frame.timestamp_ns for frame in reader] == [10, 20]


def test_readiness_requires_all_required_checks_and_ignores_optional_skip():
    checks = [CheckResult("cpu", "PASS", True, ""), CheckResult("docker", "SKIP", False, "absent")]
    assert readiness_status(checks) == "READY"
    checks.append(CheckResult("models", "SKIP", True, "missing"))
    assert readiness_status(checks) == "NOT_READY"

