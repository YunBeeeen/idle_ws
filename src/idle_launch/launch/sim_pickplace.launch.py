"""Launch sim_driver (physics + viewer) + plan_node — single-motion sim demo.

sim_driver_node owns the MuJoCo physics AND the viewer window so that mouse
perturbations (Ctrl+drag) apply forces directly to the simulated robot.

Send a target with:

    ros2 topic pub --once /ee_target_pose geometry_msgs/PoseStamped \\
        '{header: {frame_id: "world"},
          pose: {position: {x: 0.3, y: 0.0, z: 0.6},
                 orientation: {w: 1.0}}}'
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    viewer_arg = DeclareLaunchArgument(
        "viewer",
        default_value="true",
        description="Enable MuJoCo viewer window (physics-coupled, mouse perturbation enabled)",
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
    v_max_arg = DeclareLaunchArgument(
        "planner_v_max",
        default_value="1.0",
        description="Per-joint max velocity for trajectory generation (rad/s)",
    )
    a_max_arg = DeclareLaunchArgument(
        "planner_a_max",
        default_value="1.0",
        description="Per-joint max acceleration for trajectory generation (rad/s²)",
    )
    disable_gravity_arg = DeclareLaunchArgument(
        "disable_gravity",
        default_value="false",
        description="Disable gravity comp in plan_node (only for sim with dummy inertials)",
    )
    unlimited_tau_arg = DeclareLaunchArgument(
        "unlimited_tau",
        default_value="false",
        description="Bypass tau_ff clipping in plan_node (sim with dummy inertials)",
    )

    return LaunchDescription(
        [
            viewer_arg,
            viewer_left_ui_arg,
            viewer_right_ui_arg,
            v_max_arg,
            a_max_arg,
            disable_gravity_arg,
            unlimited_tau_arg,
            Node(
                package="sim",
                executable="sim_driver_node",
                name="sim_driver_node",
                output="screen",
                parameters=[
                    {
                        "viewer": LaunchConfiguration("viewer"),
                        "viewer_left_ui": LaunchConfiguration("viewer_left_ui"),
                        "viewer_right_ui": LaunchConfiguration("viewer_right_ui"),
                    }
                ],
            ),
            Node(
                package="phy",
                executable="plan_compute_node",
                name="plan_compute_node",
                output="screen",
                parameters=[
                    {
                        "planner_v_max": LaunchConfiguration("planner_v_max"),
                        "planner_a_max": LaunchConfiguration("planner_a_max"),
                    }
                ],
            ),
            Node(
                package="phy",
                executable="plan_node",
                name="plan_node",
                output="screen",
                parameters=[
                    {
                        "planner_v_max": LaunchConfiguration("planner_v_max"),
                        "planner_a_max": LaunchConfiguration("planner_a_max"),
                        "disable_gravity": LaunchConfiguration("disable_gravity"),
                        "unlimited_tau": LaunchConfiguration("unlimited_tau"),
                    }
                ],
            ),
        ]
    )
