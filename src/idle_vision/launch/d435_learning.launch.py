"""Launch D435 plus a learning-oriented image bridge."""

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
    output_color_topic_arg = DeclareLaunchArgument(
        "output_color_topic",
        default_value="/idle_vision/learning/color/image_raw",
        description="Output RGB image topic for learning.",
    )
    output_depth_topic_arg = DeclareLaunchArgument(
        "output_depth_topic",
        default_value="/idle_vision/learning/depth/image_raw",
        description="Output depth image topic for learning.",
    )
    status_topic_arg = DeclareLaunchArgument(
        "status_topic",
        default_value="/idle_vision/learning/status",
        description="JSON status topic.",
    )
    resize_width_arg = DeclareLaunchArgument("resize_width", default_value="640")
    resize_height_arg = DeclareLaunchArgument("resize_height", default_value="480")
    save_dir_arg = DeclareLaunchArgument(
        "save_dir",
        default_value="",
        description="Optional directory for saved training frames.",
    )
    save_every_n_arg = DeclareLaunchArgument(
        "save_every_n",
        default_value="0",
        description="Save every N color frames. 0 disables saving.",
    )

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

    learning_node = Node(
        package="idle_vision",
        executable="image_learning_node",
        name="image_learning_node",
        output="screen",
        parameters=[
            {
                "color_topic": LaunchConfiguration("color_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "output_color_topic": LaunchConfiguration("output_color_topic"),
                "output_depth_topic": LaunchConfiguration("output_depth_topic"),
                "status_topic": LaunchConfiguration("status_topic"),
                "resize_width": ParameterValue(
                    LaunchConfiguration("resize_width"), value_type=int
                ),
                "resize_height": ParameterValue(
                    LaunchConfiguration("resize_height"), value_type=int
                ),
                "save_dir": LaunchConfiguration("save_dir"),
                "save_every_n": ParameterValue(
                    LaunchConfiguration("save_every_n"), value_type=int
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
            output_color_topic_arg,
            output_depth_topic_arg,
            status_topic_arg,
            resize_width_arg,
            resize_height_arg,
            save_dir_arg,
            save_every_n_arg,
            camera_launch,
            learning_node,
        ]
    )
