"""Launch Logitech C270/C270i USB camera with RGB-only box pose detection."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    start_camera_arg = DeclareLaunchArgument("start_camera", default_value="true")
    start_rqt_arg = DeclareLaunchArgument("start_rqt", default_value="true")
    video_device_arg = DeclareLaunchArgument(
        "video_device",
        default_value="/dev/video2",
    )
    image_width_arg = DeclareLaunchArgument("image_width", default_value="1280")
    image_height_arg = DeclareLaunchArgument("image_height", default_value="720")
    framerate_arg = DeclareLaunchArgument("framerate", default_value="30.0")
    pixel_format_arg = DeclareLaunchArgument("pixel_format", default_value="mjpeg2rgb")
    camera_frame_id_arg = DeclareLaunchArgument(
        "camera_frame_id",
        default_value="logi_c270_frame",
    )

    target_color_arg = DeclareLaunchArgument("target_color", default_value="auto")
    hsv_ranges_json_arg = DeclareLaunchArgument("hsv_ranges_json", default_value="")
    min_area_px_arg = DeclareLaunchArgument("min_area_px", default_value="500")
    max_boxes_arg = DeclareLaunchArgument("max_boxes", default_value="20")
    use_color_ratio_mask_arg = DeclareLaunchArgument(
        "use_color_ratio_mask",
        default_value="true",
    )
    sort_by_arg = DeclareLaunchArgument("sort_by", default_value="x")
    plane_frame_arg = DeclareLaunchArgument("plane_frame", default_value="base")
    plane_z_m_arg = DeclareLaunchArgument("plane_z_m", default_value="0.0")
    plane_homography_arg = DeclareLaunchArgument(
        "plane_homography_json",
        default_value="",
        description="Optional 3x3 pixel-to-plane homography JSON.",
    )

    camera = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="logi_c270_camera",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_camera")),
        parameters=[
            {
                "video_device": LaunchConfiguration("video_device"),
                "image_width": ParameterValue(
                    LaunchConfiguration("image_width"),
                    value_type=int,
                ),
                "image_height": ParameterValue(
                    LaunchConfiguration("image_height"),
                    value_type=int,
                ),
                "framerate": ParameterValue(
                    LaunchConfiguration("framerate"),
                    value_type=float,
                ),
                "pixel_format": LaunchConfiguration("pixel_format"),
                "camera_frame_id": LaunchConfiguration("camera_frame_id"),
            }
        ],
    )

    box_pose = Node(
        package="idle_vision",
        executable="box_pose_node",
        name="logi_box_pose_node",
        output="screen",
        parameters=[
            {
                "color_topic": "/image_raw",
                "depth_topic": "/idle_vision/no_depth",
                "camera_info_topic": "/camera_info",
                "target_color": LaunchConfiguration("target_color"),
                "hsv_ranges_json": LaunchConfiguration("hsv_ranges_json"),
                "min_area_px": ParameterValue(
                    LaunchConfiguration("min_area_px"),
                    value_type=int,
                ),
                "max_boxes": ParameterValue(
                    LaunchConfiguration("max_boxes"),
                    value_type=int,
                ),
                "use_color_ratio_mask": ParameterValue(
                    LaunchConfiguration("use_color_ratio_mask"),
                    value_type=bool,
                ),
                "use_depth_candidate_mask": False,
                "require_depth": False,
                "base_frame": "",
                "sort_by": LaunchConfiguration("sort_by"),
                "plane_frame": LaunchConfiguration("plane_frame"),
                "plane_z_m": ParameterValue(
                    LaunchConfiguration("plane_z_m"),
                    value_type=float,
                ),
                "plane_homography_json": LaunchConfiguration("plane_homography_json"),
            }
        ],
    )

    debug_view = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="rqt_logi_box_debug",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_rqt")),
        arguments=["/idle_vision/box_pose/debug_image"],
    )
    color_view = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="rqt_logi_color",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_rqt")),
        arguments=["/image_raw"],
    )

    return LaunchDescription(
        [
            start_camera_arg,
            start_rqt_arg,
            video_device_arg,
            image_width_arg,
            image_height_arg,
            framerate_arg,
            pixel_format_arg,
            camera_frame_id_arg,
            target_color_arg,
            hsv_ranges_json_arg,
            min_area_px_arg,
            max_boxes_arg,
            use_color_ratio_mask_arg,
            sort_by_arg,
            plane_frame_arg,
            plane_z_m_arg,
            plane_homography_arg,
            camera,
            box_pose,
            debug_view,
            color_view,
        ]
    )
