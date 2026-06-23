"""Launch hold controller with MuJoCo viewer mirroring motor state."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    viewer_arg = DeclareLaunchArgument(
        "viewer",
        default_value="true",
        description="Enable MuJoCo viewer window",
    )
    viewer_left_ui_arg = DeclareLaunchArgument(
        "viewer_left_ui",
        default_value="true",
        description="Show MuJoCo left UI panel",
    )
    viewer_right_ui_arg = DeclareLaunchArgument(
        "viewer_right_ui",
        default_value="true",
        description="Show MuJoCo right UI panel",
    )

    return LaunchDescription(
        [
            viewer_arg,
            viewer_left_ui_arg,
            viewer_right_ui_arg,
            Node(
                package="phy",
                executable="hold_node",
                name="hold_node",
                output="screen",
            ),
            Node(
                package="sim",
                executable="viewer_node",
                name="viewer_node",
                output="screen",
                parameters=[
                    {
                        "viewer": LaunchConfiguration("viewer"),
                        "viewer_left_ui": LaunchConfiguration("viewer_left_ui"),
                        "viewer_right_ui": LaunchConfiguration("viewer_right_ui"),
                    }
                ],
            ),
        ]
    )
