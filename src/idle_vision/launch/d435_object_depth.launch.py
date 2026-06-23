"""Launch D435 and extract depth for one target-colored object."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    start_camera_arg = DeclareLaunchArgument(
        "start_camera",
        default_value="true",
        description="Start the RealSense driver from this launch file.",
    )
    camera_name_arg = DeclareLaunchArgument("camera_name", default_value="camera")
    camera_namespace_arg = DeclareLaunchArgument("camera_namespace", default_value="camera")
    serial_no_arg = DeclareLaunchArgument("serial_no", default_value="''")
    usb_port_id_arg = DeclareLaunchArgument("usb_port_id", default_value="''")
    device_type_arg = DeclareLaunchArgument("device_type", default_value="d435")
    color_profile_arg = DeclareLaunchArgument("color_profile", default_value="640,480,30")
    depth_profile_arg = DeclareLaunchArgument("depth_profile", default_value="640,480,30")
    initial_reset_arg = DeclareLaunchArgument("initial_reset", default_value="false")

    color_topic_arg = DeclareLaunchArgument(
        "color_topic",
        default_value="/camera/camera/color/image_raw",
        description="Input RGB image topic.",
    )
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/camera/camera/aligned_depth_to_color/image_raw",
        description="Input aligned depth topic.",
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/camera/color/camera_info",
        description="Input color camera info topic.",
    )
    target_color_arg = DeclareLaunchArgument(
        "target_color",
        default_value="auto",
        description=(
            "HSV preset or auto. auto scans strict red, green, blue. "
            "Use all to include white/black too."
        ),
    )
    hsv_ranges_json_arg = DeclareLaunchArgument(
        "hsv_ranges_json",
        default_value="",
        description="Optional custom HSV ranges JSON.",
    )
    min_area_px_arg = DeclareLaunchArgument("min_area_px", default_value="500")
    max_area_px_arg = DeclareLaunchArgument("max_area_px", default_value="0")
    min_depth_m_arg = DeclareLaunchArgument("min_depth_m", default_value="0.05")
    max_depth_m_arg = DeclareLaunchArgument("max_depth_m", default_value="5.0")
    depth_percentile_arg = DeclareLaunchArgument("depth_percentile", default_value="50.0")
    use_color_ratio_mask_arg = DeclareLaunchArgument(
        "use_color_ratio_mask",
        default_value="true",
        description="Combine HSV mask with normalized BGR channel ratio checks.",
    )
    use_depth_candidate_mask_arg = DeclareLaunchArgument(
        "use_depth_candidate_mask",
        default_value="true",
        description="Intersect candidate color masks with valid depth pixels.",
    )
    publish_debug_arg = DeclareLaunchArgument("publish_debug", default_value="true")

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("idle_vision"), "launch", "d435_camera.launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("start_camera")),
        launch_arguments={
            "camera_name": LaunchConfiguration("camera_name"),
            "camera_namespace": LaunchConfiguration("camera_namespace"),
            "serial_no": LaunchConfiguration("serial_no"),
            "usb_port_id": LaunchConfiguration("usb_port_id"),
            "device_type": LaunchConfiguration("device_type"),
            "color_profile": LaunchConfiguration("color_profile"),
            "depth_profile": LaunchConfiguration("depth_profile"),
            "align_depth": "true",
            "enable_sync": "true",
            "initial_reset": LaunchConfiguration("initial_reset"),
        }.items(),
    )

    object_depth_node = Node(
        package="idle_vision",
        executable="object_depth_node",
        name="object_depth_node",
        output="screen",
        parameters=[
            {
                "color_topic": LaunchConfiguration("color_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "target_color": LaunchConfiguration("target_color"),
                "hsv_ranges_json": LaunchConfiguration("hsv_ranges_json"),
                "min_area_px": ParameterValue(
                    LaunchConfiguration("min_area_px"), value_type=int
                ),
                "max_area_px": ParameterValue(
                    LaunchConfiguration("max_area_px"), value_type=int
                ),
                "min_depth_m": ParameterValue(
                    LaunchConfiguration("min_depth_m"), value_type=float
                ),
                "max_depth_m": ParameterValue(
                    LaunchConfiguration("max_depth_m"), value_type=float
                ),
                "depth_percentile": ParameterValue(
                    LaunchConfiguration("depth_percentile"), value_type=float
                ),
                "use_color_ratio_mask": ParameterValue(
                    LaunchConfiguration("use_color_ratio_mask"), value_type=bool
                ),
                "use_depth_candidate_mask": ParameterValue(
                    LaunchConfiguration("use_depth_candidate_mask"), value_type=bool
                ),
                "publish_debug": ParameterValue(
                    LaunchConfiguration("publish_debug"), value_type=bool
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            start_camera_arg,
            camera_name_arg,
            camera_namespace_arg,
            serial_no_arg,
            usb_port_id_arg,
            device_type_arg,
            color_profile_arg,
            depth_profile_arg,
            initial_reset_arg,
            color_topic_arg,
            depth_topic_arg,
            camera_info_topic_arg,
            target_color_arg,
            hsv_ranges_json_arg,
            min_area_px_arg,
            max_area_px_arg,
            min_depth_m_arg,
            max_depth_m_arg,
            depth_percentile_arg,
            use_color_ratio_mask_arg,
            use_depth_candidate_mask_arg,
            publish_debug_arg,
            camera_launch,
            object_depth_node,
        ]
    )
