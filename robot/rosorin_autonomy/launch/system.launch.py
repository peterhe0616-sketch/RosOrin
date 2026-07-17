"""Bring up MentorPi hardware first, then the ROSOrin autonomy stack."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def package_launch(name: str) -> IncludeLaunchDescription:
    path = Path(get_package_share_directory("rosorin_autonomy")) / "launch" / name
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(str(path)))


def generate_launch_description():
    return LaunchDescription(
        [
            package_launch("minimal_robot.launch.py"),
            # The MentorPi IMU performs an initialization/calibration cycle before
            # its filtered odometry and TF become reliable.  Give the hardware
            # stack enough time before Nav2 starts checking the odom frame.
            TimerAction(period=15.0, actions=[package_launch("autonomy.launch.py")]),
        ]
    )
