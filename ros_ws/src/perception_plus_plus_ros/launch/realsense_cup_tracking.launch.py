import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("perception_plus_plus_ros"))
    realsense_share = Path(get_package_share_directory("realsense2_camera"))
    project_root = LaunchConfiguration("project_root")
    parameters_file = LaunchConfiguration("parameters_file")
    arguments = [
        DeclareLaunchArgument("project_root", default_value=os.getcwd()),
        DeclareLaunchArgument(
            "parameters_file",
            default_value=str(package_share / "config" / "cup_tracking.yaml")),
    ]
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(realsense_share / "launch" / "rs_launch.py")),
        launch_arguments={
            "align_depth.enable": "true",
            "enable_sync": "true",
            "rgb_camera.color_profile": "640x480x30",
            "depth_module.depth_profile": "640x480x30",
        }.items(),
    )
    tracker = Node(
        package="perception_plus_plus_ros",
        executable="cup_tracking_node",
        name="cup_tracking",
        output="screen",
        parameters=[parameters_file, {"project_root": project_root}],
    )
    return LaunchDescription(arguments + [camera, tracker])
