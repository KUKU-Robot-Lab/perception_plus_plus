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


def test_msgs_package_declares_ament_cmake_build_type():
    # Without the explicit export colcon identifies the package as ros.catkin,
    # which installs no AMENT_PREFIX_PATH hook and hides the messages from
    # `ros2 interface show` in the container.
    text = (ROOT / "perception_plus_plus_msgs/package.xml").read_text()
    assert "<build_type>ament_cmake</build_type>" in text


def test_ros_defaults_use_yolo_segmentation_weights():
    config = (ROOT / "perception_plus_plus_ros/config/cup_tracking.yaml").read_text()
    node = (ROOT / "perception_plus_plus_ros/perception_plus_plus_ros/node.py").read_text()
    assert "models/yolo/yolo11n-seg.pt" in config
    assert "models/yolo/yolo11n-seg.pt" in node


def test_combined_launch_uses_validated_realsense_profile_and_sync():
    launch = (
        ROOT / "perception_plus_plus_ros/launch/realsense_cup_tracking.launch.py"
    )
    assert launch.is_file()
    text = launch.read_text()
    for token in (
        "640x480x30",
        "align_depth.enable",
        "enable_sync",
    ):
        assert token in text
