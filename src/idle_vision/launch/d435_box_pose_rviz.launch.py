"""Launch D435 box pose pipeline with RViz visualization."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    start_rviz_arg = DeclareLaunchArgument(
        "start_rviz",
        default_value="true",
        description="Start RViz with the box pose layout.",
    )
    start_camera_arg = DeclareLaunchArgument("start_camera", default_value="true")
    camera_publish_tf_arg = DeclareLaunchArgument("camera_publish_tf", default_value="false")
    target_color_arg = DeclareLaunchArgument("target_color", default_value="auto")
    hsv_ranges_json_arg = DeclareLaunchArgument("hsv_ranges_json", default_value="")
    min_area_px_arg = DeclareLaunchArgument("min_area_px", default_value="500")
    max_boxes_arg = DeclareLaunchArgument("max_boxes", default_value="20")
    use_color_ratio_mask_arg = DeclareLaunchArgument(
        "use_color_ratio_mask",
        default_value="true",
    )
    use_depth_candidate_mask_arg = DeclareLaunchArgument(
        "use_depth_candidate_mask",
        default_value="true",
    )
    sort_by_arg = DeclareLaunchArgument("sort_by", default_value="x")
    base_frame_arg = DeclareLaunchArgument("base_frame", default_value="base")
    camera_frame_arg = DeclareLaunchArgument(
        "camera_frame",
        default_value="camera_color_optical_frame",
        description="Camera frame to attach to base.",
    )
    publish_camera_tf_arg = DeclareLaunchArgument(
        "publish_camera_tf",
        default_value="true",
        description="Publish a static base->camera calibration transform.",
    )
    camera_x_arg = DeclareLaunchArgument("camera_x", default_value="0.067")
    camera_y_arg = DeclareLaunchArgument("camera_y", default_value="0.56")
    camera_z_arg = DeclareLaunchArgument("camera_z", default_value="0.939")
    camera_roll_arg = DeclareLaunchArgument("camera_roll", default_value="3.141592")
    camera_pitch_arg = DeclareLaunchArgument("camera_pitch", default_value="0.0")
    camera_yaw_arg = DeclareLaunchArgument("camera_yaw", default_value="3.141592")

    box_pose_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("idle_vision"), "launch", "d435_box_pose.launch.py"]
            )
        ),
        launch_arguments={
            "start_camera": LaunchConfiguration("start_camera"),
            "camera_publish_tf": LaunchConfiguration("camera_publish_tf"),
            "target_color": LaunchConfiguration("target_color"),
            "hsv_ranges_json": LaunchConfiguration("hsv_ranges_json"),
            "min_area_px": LaunchConfiguration("min_area_px"),
            "max_boxes": LaunchConfiguration("max_boxes"),
            "use_color_ratio_mask": LaunchConfiguration("use_color_ratio_mask"),
            "use_depth_candidate_mask": LaunchConfiguration("use_depth_candidate_mask"),
            "sort_by": LaunchConfiguration("sort_by"),
            "base_frame": LaunchConfiguration("base_frame"),
        }.items(),
    )

    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_to_base_static_tf",
        output="screen",
        condition=IfCondition(LaunchConfiguration("publish_camera_tf")),
        arguments=[
            "--x",
            LaunchConfiguration("camera_x"),
            "--y",
            LaunchConfiguration("camera_y"),
            "--z",
            LaunchConfiguration("camera_z"),
            "--yaw",
            LaunchConfiguration("camera_yaw"),
            "--pitch",
            LaunchConfiguration("camera_pitch"),
            "--roll",
            LaunchConfiguration("camera_roll"),
            "--frame-id",
            LaunchConfiguration("base_frame"),
            "--child-frame-id",
            LaunchConfiguration("camera_frame"),
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_rviz")),
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("idle_vision"), "rviz", "d435_box_pose.rviz"]
            ),
        ],
    )

    return LaunchDescription(
        [
            start_rviz_arg,
            start_camera_arg,
            camera_publish_tf_arg,
            target_color_arg,
            hsv_ranges_json_arg,
            min_area_px_arg,
            max_boxes_arg,
            use_color_ratio_mask_arg,
            use_depth_candidate_mask_arg,
            sort_by_arg,
            base_frame_arg,
            camera_frame_arg,
            publish_camera_tf_arg,
            camera_x_arg,
            camera_y_arg,
            camera_z_arg,
            camera_roll_arg,
            camera_pitch_arg,
            camera_yaw_arg,
            camera_tf,
            box_pose_launch,
            rviz,
        ]
    )
