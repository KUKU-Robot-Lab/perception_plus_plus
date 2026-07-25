from pathlib import Path


ROOT = Path("ros_ws/src")


def test_tracking_status_contract_has_required_fields_and_states():
    text = (ROOT / "perception_plus_plus_msgs/msg/TrackingStatus.msg").read_text()
    for token in ("INITIALIZING", "TRACKING", "LOST", "REINITIALIZING",
                  "last_valid_pose_stamp", "failure_reason", "consecutive_valid",
                  "consecutive_invalid", "fatal"):
        assert token in text


def test_ros_node_uses_three_way_sync_and_pose_gate():
    text = (ROOT / "perception_plus_plus_ros/perception_plus_plus_ros/node.py").read_text()
    assert "ApproximateTimeSynchronizer" in text
    assert "if output.pose is not None" in text
    assert "sendTransform" in text


def test_humble_and_jazzy_share_ros_sources():
    assert not (Path("docker/humble") / "ros_ws").exists()
    assert not (Path("docker/jazzy") / "ros_ws").exists()

