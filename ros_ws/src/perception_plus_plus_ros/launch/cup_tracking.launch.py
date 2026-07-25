from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("parameters_file", default_value=str(
            Path(__file__).parents[1] / "config" / "cup_tracking.yaml")),
    ]
    node = Node(
        package="perception_plus_plus_ros",
        executable="cup_tracking_node",
        name="cup_tracking",
        output="screen",
        parameters=[LaunchConfiguration("parameters_file")],
    )
    return LaunchDescription(arguments + [node])

