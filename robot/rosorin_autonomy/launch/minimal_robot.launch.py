"""Start only the MentorPi hardware services needed by autonomous navigation."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def include(package: str, launch_file: str) -> IncludeLaunchDescription:
    path = Path(get_package_share_directory(package)) / "launch" / launch_file
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(str(path)))


def generate_launch_description():
    return LaunchDescription(
        [
            include("controller", "controller.launch.py"),
            include("peripherals", "depth_camera.launch.py"),
            include("peripherals", "lidar.launch.py"),
            ExecuteProcess(
                cmd=["ros2", "launch", "rosbridge_server", "rosbridge_websocket_launch.xml"],
                output="screen",
            ),
            Node(
                package="web_video_server",
                executable="web_video_server",
                name="web_video_server",
                output="screen",
            ),
        ]
    )

