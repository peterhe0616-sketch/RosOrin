from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def generate_launch_description():
    package_dir = Path(get_package_share_directory("rosorin_autonomy"))
    nav2_dir = Path(get_package_share_directory("nav2_bringup"))
    params_file = LaunchConfiguration("params_file")
    slam_params_file = LaunchConfiguration("slam_params_file")
    collision_params_file = LaunchConfiguration("collision_params_file")

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(get_package_share_directory("slam_toolbox")) / "launch" / "online_async_launch.py")
        ),
        launch_arguments={
            "use_sim_time": "false",
            "slam_params_file": slam_params_file,
        }.items(),
    )

    nav = GroupAction(
        actions=[
            SetRemap(src="cmd_vel", dst="/autonomy/cmd_vel_smoothed"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(nav2_dir / "launch" / "navigation_launch.py")
                ),
                launch_arguments={
                    "use_sim_time": "false",
                    "autostart": "true",
                    "use_composition": "false",
                    "params_file": params_file,
                }.items(),
            ),
        ]
    )

    collision_monitor = Node(
        package="nav2_collision_monitor",
        executable="collision_monitor",
        name="collision_monitor",
        output="screen",
        parameters=[collision_params_file],
    )
    map_saver = Node(
        package="nav2_map_server",
        executable="map_saver_server",
        name="map_saver",
        output="screen",
        parameters=[params_file],
    )
    collision_lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_collision",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "autostart": True,
                "node_names": ["collision_monitor", "map_saver"],
            }
        ],
    )
    bridge = Node(
        package="rosorin_autonomy",
        executable="autonomy_bridge",
        name="autonomy_bridge",
        output="screen",
        parameters=[
            {
                "status_rate_hz": 2.0,
                "stuck_timeout_s": 2.5,
                "movement_epsilon_m": 0.025,
                "rotation_epsilon_rad": 0.08,
                "map_directory": "/home/ubuntu/shared/maps",
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file", default_value=str(package_dir / "config" / "nav2.yaml")
            ),
            DeclareLaunchArgument(
                "slam_params_file", default_value=str(package_dir / "config" / "slam.yaml")
            ),
            DeclareLaunchArgument(
                "collision_params_file",
                default_value=str(package_dir / "config" / "collision_monitor.yaml"),
            ),
            slam,
            nav,
            collision_monitor,
            map_saver,
            collision_lifecycle,
            bridge,
        ]
    )
