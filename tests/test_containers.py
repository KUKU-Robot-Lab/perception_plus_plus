from pathlib import Path


def test_distribution_specific_containers_share_source_contract():
    humble = Path("docker/humble/Dockerfile").read_text()
    jazzy = Path("docker/jazzy/Dockerfile").read_text()
    assert "ubuntu22.04" in humble and "ROS_DISTRO=humble" in humble
    assert "ubuntu24.04" in jazzy and "ROS_DISTRO=jazzy" in jazzy
    for text in (humble, jazzy):
        assert "ros_ws/src" in text
        assert "USER perception" in text
        assert "NVIDIA_VISIBLE_DEVICES" in text


def test_lock_files_are_distribution_specific():
    humble = Path("docker/humble/requirements.lock").read_text()
    jazzy = Path("docker/jazzy/requirements.lock").read_text()
    assert humble != jazzy
    assert "torch==" in humble and "torch==" in jazzy

