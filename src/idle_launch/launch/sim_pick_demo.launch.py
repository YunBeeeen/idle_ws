"""Launch sim_driver + plan_node for pick-and-place demo.

Uses plan_node with elbow_up_filter=true and hybrid fold-and-rotate motion.
Per-joint v_max/a_max/kp are read from control_params.yaml at runtime.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("viewer",           default_value="true"),
        DeclareLaunchArgument("viewer_left_ui",   default_value="true"),
        DeclareLaunchArgument("viewer_right_ui",  default_value="true"),
        DeclareLaunchArgument("planner_v_max",    default_value="1.0"),
        DeclareLaunchArgument("planner_a_max",    default_value="1.0"),
        DeclareLaunchArgument("planner_min_traj_duration", default_value="1.5"),
        DeclareLaunchArgument("disable_gravity",  default_value="false"),
        DeclareLaunchArgument("unlimited_tau",    default_value="true"),
        DeclareLaunchArgument("warp_q_hi_rad",    default_value="1.0"),

        Node(
            package="sim",
            executable="sim_driver_node",
            name="sim_driver_node",
            output="screen",
            parameters=[{
                "viewer":          LaunchConfiguration("viewer"),
                "viewer_left_ui":  LaunchConfiguration("viewer_left_ui"),
                "viewer_right_ui": LaunchConfiguration("viewer_right_ui"),
            }],
        ),
        Node(
            package="phy",
            executable="plan_node",
            name="plan_node",
            output="screen",
            parameters=[{
                "planner_v_max":             LaunchConfiguration("planner_v_max"),
                "planner_a_max":             LaunchConfiguration("planner_a_max"),
                "planner_min_traj_duration": LaunchConfiguration("planner_min_traj_duration"),
                "disable_gravity":           LaunchConfiguration("disable_gravity"),
                "unlimited_tau":             LaunchConfiguration("unlimited_tau"),
                "warp_q_hi_rad":             LaunchConfiguration("warp_q_hi_rad"),
            }],
        ),
    ])
