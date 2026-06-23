"""Launch Intel RealSense D435 with RGB, depth, and aligned depth streams."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    camera_name_arg = DeclareLaunchArgument(
        "camera_name",
        default_value="camera",
        description="RealSense node name",
    )
    camera_namespace_arg = DeclareLaunchArgument(
        "camera_namespace",
        default_value="camera",
        description="RealSense namespace",
    )
    serial_no_arg = DeclareLaunchArgument(
        "serial_no",
        default_value="''",
        description="Optional RealSense serial number. Use _SERIAL or quoted SERIAL.",
    )
    usb_port_id_arg = DeclareLaunchArgument(
        "usb_port_id",
        default_value="''",
        description="Optional USB port id selector.",
    )
    device_type_arg = DeclareLaunchArgument(
        "device_type",
        default_value="d435",
        description="RealSense device type selector.",
    )
    color_profile_arg = DeclareLaunchArgument(
        "color_profile",
        default_value="640,480,30",
        description="Color profile as width,height,fps.",
    )
    depth_profile_arg = DeclareLaunchArgument(
        "depth_profile",
        default_value="640,480,30",
        description="Depth profile as width,height,fps.",
    )
    align_depth_arg = DeclareLaunchArgument(
        "align_depth",
        default_value="true",
        description="Publish depth aligned to color.",
    )
    enable_sync_arg = DeclareLaunchArgument(
        "enable_sync",
        default_value="true",
        description="Synchronize color/depth frames in the RealSense driver.",
    )
    pointcloud_arg = DeclareLaunchArgument(
        "pointcloud",
        default_value="false",
        description="Enable RealSense point cloud topic.",
    )
    publish_tf_arg = DeclareLaunchArgument(
        "publish_tf",
        default_value="true",
        description="Publish RealSense internal TF frames.",
    )
    initial_reset_arg = DeclareLaunchArgument(
        "initial_reset",
        default_value="false",
        description="Reset camera on start.",
    )
    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="RealSense log level.",
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments={
            "camera_name": LaunchConfiguration("camera_name"),
            "camera_namespace": LaunchConfiguration("camera_namespace"),
            "serial_no": LaunchConfiguration("serial_no"),
            "usb_port_id": LaunchConfiguration("usb_port_id"),
            "device_type": LaunchConfiguration("device_type"),
            "enable_color": "true",
            "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
            "rgb_camera.color_format": "RGB8",
            "enable_depth": "true",
            "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
            "depth_module.depth_format": "Z16",
            "align_depth.enable": LaunchConfiguration("align_depth"),
            "enable_sync": LaunchConfiguration("enable_sync"),
            "pointcloud.enable": LaunchConfiguration("pointcloud"),
            "enable_infra1": "false",
            "enable_infra2": "false",
            "initial_reset": LaunchConfiguration("initial_reset"),
            "publish_tf": LaunchConfiguration("publish_tf"),
            "log_level": LaunchConfiguration("log_level"),
        }.items(),
    )

    return LaunchDescription(
        [
            camera_name_arg,
            camera_namespace_arg,
            serial_no_arg,
            usb_port_id_arg,
            device_type_arg,
            color_profile_arg,
            depth_profile_arg,
            align_depth_arg,
            enable_sync_arg,
            pointcloud_arg,
            publish_tf_arg,
            initial_reset_arg,
            log_level_arg,
            realsense_launch,
        ]
    )
