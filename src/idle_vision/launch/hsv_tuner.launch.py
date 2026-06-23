"""Launch D435 and an HSV ROI inspector for threshold tuning."""

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
    sample_u_arg = DeclareLaunchArgument(
        "sample_u",
        default_value="-1",
        description="Sample pixel x. -1 uses image center.",
    )
    sample_v_arg = DeclareLaunchArgument(
        "sample_v",
        default_value="-1",
        description="Sample pixel y. -1 uses image center.",
    )
    roi_half_size_px_arg = DeclareLaunchArgument(
        "roi_half_size_px",
        default_value="12",
        description="Half-size of HSV sample ROI in pixels.",
    )
    publish_every_n_arg = DeclareLaunchArgument(
        "publish_every_n",
        default_value="5",
        description="Publish HSV stats every N color frames.",
    )
    hsv_ranges_json_arg = DeclareLaunchArgument(
        "hsv_ranges_json",
        default_value="",
        description="Optional HSV range to preview mask output.",
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
            "initial_reset": LaunchConfiguration("initial_reset"),
        }.items(),
    )

    tuner_node = Node(
        package="idle_vision",
        executable="hsv_tuner_node",
        name="hsv_tuner_node",
        output="screen",
        parameters=[
            {
                "color_topic": LaunchConfiguration("color_topic"),
                "sample_u": ParameterValue(LaunchConfiguration("sample_u"), value_type=int),
                "sample_v": ParameterValue(LaunchConfiguration("sample_v"), value_type=int),
                "roi_half_size_px": ParameterValue(
                    LaunchConfiguration("roi_half_size_px"), value_type=int
                ),
                "publish_every_n": ParameterValue(
                    LaunchConfiguration("publish_every_n"), value_type=int
                ),
                "hsv_ranges_json": LaunchConfiguration("hsv_ranges_json"),
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
            sample_u_arg,
            sample_v_arg,
            roi_half_size_px_arg,
            publish_every_n_arg,
            hsv_ranges_json_arg,
            publish_debug_arg,
            camera_launch,
            tuner_node,
        ]
    )
