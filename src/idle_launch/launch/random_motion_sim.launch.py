"""Launch MuJoCo sim + viewer + random joint_sweep_node motion preview.

This is the hardware-free preview path for no-contact random free-motion data
collection. It uses the exact same ``joint_sweep_node`` motion generator as the
real robot path, but replaces CAN with ``sim_driver_node``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    viewer_arg = DeclareLaunchArgument(
        "viewer",
        default_value="false",
        description="Enable MuJoCo viewer window. Keep false on machines with GLX viewer issues.",
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
    viewer_software_gl_arg = DeclareLaunchArgument(
        "viewer_software_gl",
        default_value="false",
        description="Force Mesa software GL for MuJoCo viewer when GLX context creation fails",
    )
    viewer_hz_arg = DeclareLaunchArgument(
        "viewer_hz",
        default_value="20.0",
        description="MuJoCo viewer refresh rate. Lower values reduce GUI lag.",
    )
    sim_control_hz_arg = DeclareLaunchArgument(
        "sim_control_hz",
        default_value="125.0",
        description="MuJoCo sim driver publish/control rate.",
    )
    command_hz_arg = DeclareLaunchArgument(
        "command_hz",
        default_value="100.0",
        description="joint_sweep_node command publish rate.",
    )
    random_seed_arg = DeclareLaunchArgument(
        "random_seed",
        default_value="7",
        description="Random waypoint seed for reproducible no-contact motion",
    )
    random_waypoint_count_arg = DeclareLaunchArgument(
        "random_waypoint_count",
        default_value="8",
        description="Number of random waypoints to generate",
    )
    random_range_arg = DeclareLaunchArgument(
        "random_range_by_motor_json",
        default_value='{"2": [-1.20, -0.45], "3": [-1.55, -0.45]}',
        description="Per-motor random sampling range in radians",
    )
    waypoints_path_arg = DeclareLaunchArgument(
        "waypoints_path",
        default_value="",
        description="Optional explicit joint waypoint JSON path. If set, joint_sweep_node uses it instead of generated random waypoints.",
    )
    sine_motion_arg = DeclareLaunchArgument(
        "sine_motion_enabled",
        default_value="false",
        description="Use continuous sine free-motion instead of random/waypoint motion.",
    )
    sine_center_arg = DeclareLaunchArgument(
        "sine_center_by_motor_json",
        default_value='{"2": -0.70, "3": -1.30}',
        description="Sine center angle by motor id.",
    )
    sine_amplitude_arg = DeclareLaunchArgument(
        "sine_amplitude_by_motor_json",
        default_value='{"2": 0.08, "3": 0.08}',
        description="Sine amplitude by motor id.",
    )
    sine_frequency_arg = DeclareLaunchArgument(
        "sine_frequency_by_motor_json",
        default_value='{"2": 0.025, "3": 0.040}',
        description="Sine frequency in Hz by motor id.",
    )
    sine_phase_arg = DeclareLaunchArgument(
        "sine_phase_by_motor_json",
        default_value='{"2": 0.0, "3": 1.57079632679}',
        description="Sine phase in radians by motor id.",
    )
    sine_envelope_period_arg = DeclareLaunchArgument(
        "sine_envelope_period_s",
        default_value="80.0",
        description="Amplitude envelope grow/shrink period.",
    )
    sine_duration_arg = DeclareLaunchArgument(
        "sine_duration_s",
        default_value="0.0",
        description="Sine duration. 0 means run until interrupted.",
    )
    segment_duration_arg = DeclareLaunchArgument(
        "segment_duration_s",
        default_value="18.0",
        description="Duration of each random waypoint segment",
    )
    initial_duration_arg = DeclareLaunchArgument(
        "initial_to_zero_duration_s",
        default_value="12.0",
        description="Duration from initial pose to first waypoint",
    )
    repeat_arg = DeclareLaunchArgument(
        "repeat",
        default_value="false",
        description="Repeat the generated waypoint sequence",
    )
    min_floor_clearance_arg = DeclareLaunchArgument(
        "min_floor_clearance_m",
        default_value="0.02",
        description="Minimum allowed mesh clearance above floor z=0",
    )

    return LaunchDescription(
        [
            viewer_arg,
            viewer_left_ui_arg,
            viewer_right_ui_arg,
            viewer_software_gl_arg,
            viewer_hz_arg,
            sim_control_hz_arg,
            command_hz_arg,
            SetEnvironmentVariable(
                "LIBGL_ALWAYS_SOFTWARE",
                "1",
                condition=IfCondition(LaunchConfiguration("viewer_software_gl")),
            ),
            SetEnvironmentVariable(
                "__GLX_VENDOR_LIBRARY_NAME",
                "mesa",
                condition=IfCondition(LaunchConfiguration("viewer_software_gl")),
            ),
            SetEnvironmentVariable(
                "MESA_GL_VERSION_OVERRIDE",
                "3.3",
                condition=IfCondition(LaunchConfiguration("viewer_software_gl")),
            ),
            random_seed_arg,
            random_waypoint_count_arg,
            random_range_arg,
            waypoints_path_arg,
            sine_motion_arg,
            sine_center_arg,
            sine_amplitude_arg,
            sine_frequency_arg,
            sine_phase_arg,
            sine_envelope_period_arg,
            sine_duration_arg,
            segment_duration_arg,
            initial_duration_arg,
            repeat_arg,
            min_floor_clearance_arg,
            Node(
                package="sim",
                executable="sim_driver_node",
                name="sim_driver_node",
                output="screen",
                parameters=[
                    {
                        "sim_control_hz": ParameterValue(
                            LaunchConfiguration("sim_control_hz"),
                            value_type=float,
                        ),
                    }
                ],
            ),
            Node(
                package="sim",
                executable="viewer_node",
                name="viewer_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("viewer")),
                parameters=[
                    {
                        "viewer": ParameterValue(LaunchConfiguration("viewer"), value_type=bool),
                        "viewer_hz": ParameterValue(
                            LaunchConfiguration("viewer_hz"),
                            value_type=float,
                        ),
                        "viewer_left_ui": ParameterValue(
                            LaunchConfiguration("viewer_left_ui"),
                            value_type=bool,
                        ),
                        "viewer_right_ui": ParameterValue(
                            LaunchConfiguration("viewer_right_ui"),
                            value_type=bool,
                        ),
                    }
                ],
            ),
            Node(
                package="phy",
                executable="joint_sweep_node",
                name="joint_sweep_node",
                output="screen",
                parameters=[
                    {
                        "random_motion_enabled": True,
                        "sine_motion_enabled": ParameterValue(
                            LaunchConfiguration("sine_motion_enabled"),
                            value_type=bool,
                        ),
                        "sine_center_by_motor_json": ParameterValue(
                            LaunchConfiguration("sine_center_by_motor_json"),
                            value_type=str,
                        ),
                        "sine_amplitude_by_motor_json": ParameterValue(
                            LaunchConfiguration("sine_amplitude_by_motor_json"),
                            value_type=str,
                        ),
                        "sine_frequency_by_motor_json": ParameterValue(
                            LaunchConfiguration("sine_frequency_by_motor_json"),
                            value_type=str,
                        ),
                        "sine_phase_by_motor_json": ParameterValue(
                            LaunchConfiguration("sine_phase_by_motor_json"),
                            value_type=str,
                        ),
                        "sine_envelope_period_s": ParameterValue(
                            LaunchConfiguration("sine_envelope_period_s"),
                            value_type=float,
                        ),
                        "sine_duration_s": ParameterValue(
                            LaunchConfiguration("sine_duration_s"),
                            value_type=float,
                        ),
                        "control_hz": ParameterValue(
                            LaunchConfiguration("command_hz"),
                            value_type=float,
                        ),
                        "random_motor_ids_json": [2, 3],
                        "random_range_by_motor_json": ParameterValue(
                            LaunchConfiguration("random_range_by_motor_json"),
                            value_type=str,
                        ),
                        "waypoints_path": ParameterValue(
                            LaunchConfiguration("waypoints_path"),
                            value_type=str,
                        ),
                        "random_waypoint_count": ParameterValue(
                            LaunchConfiguration("random_waypoint_count"),
                            value_type=int,
                        ),
                        "random_seed": ParameterValue(
                            LaunchConfiguration("random_seed"),
                            value_type=int,
                        ),
                        "segment_duration_s": ParameterValue(
                            LaunchConfiguration("segment_duration_s"),
                            value_type=float,
                        ),
                        "initial_to_zero_duration_s": ParameterValue(
                            LaunchConfiguration("initial_to_zero_duration_s"),
                            value_type=float,
                        ),
                        "repeat": ParameterValue(
                            LaunchConfiguration("repeat"),
                            value_type=bool,
                        ),
                        "sweep_kp": 4.0,
                        "sweep_kd": 0.8,
                        "hold_kp": 4.0,
                        "hold_kd": 0.8,
                        "enforce_joint_limits": True,
                        "enforce_zero_crossing": True,
                        "check_self_collision": True,
                        "check_floor_clearance": True,
                        "min_floor_clearance_m": ParameterValue(
                            LaunchConfiguration("min_floor_clearance_m"),
                            value_type=float,
                        ),
                        "check_gravity_load": True,
                        "gravity_load_limit_by_motor_json": ParameterValue(
                            '{"2": 25.0, "3": 11.0}',
                            value_type=str,
                        ),
                        "collision_samples_per_segment": 8,
                        "random_max_attempts": 250,
                        "random_min_step_norm_rad": 0.08,
                        "random_max_step_norm_rad": 0.45,
                    }
                ],
            ),
        ]
    )
