"""Launch D435 box pose detection plus an Ollama/Qwen command selector."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    start_rqt_arg = DeclareLaunchArgument("start_rqt", default_value="true")
    start_camera_arg = DeclareLaunchArgument("start_camera", default_value="true")
    start_voice_arg = DeclareLaunchArgument("start_voice", default_value="false")
    use_ollama_arg = DeclareLaunchArgument("use_ollama", default_value="true")
    ollama_model_arg = DeclareLaunchArgument("ollama_model", default_value="qwen2.5:7b")
    ollama_url_arg = DeclareLaunchArgument(
        "ollama_url",
        default_value="http://127.0.0.1:11434/api/generate",
    )
    default_spatial_arg = DeclareLaunchArgument("default_spatial", default_value="largest")
    spatial_reference_arg = DeclareLaunchArgument("spatial_reference", default_value="image")
    whisper_model_arg = DeclareLaunchArgument("whisper_model_size", default_value="small")
    whisper_device_arg = DeclareLaunchArgument("whisper_device", default_value="cpu")
    whisper_compute_arg = DeclareLaunchArgument("whisper_compute_type", default_value="int8")
    input_device_arg = DeclareLaunchArgument("input_device", default_value="")
    energy_multiplier_arg = DeclareLaunchArgument("energy_multiplier", default_value="1.4")
    min_threshold_arg = DeclareLaunchArgument("min_absolute_threshold", default_value="300.0")
    log_energy_arg = DeclareLaunchArgument("log_energy", default_value="false")

    box_pose_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("idle_vision"),
                    "launch",
                    "d435_box_pose_rqt.launch.py",
                ]
            )
        ),
        launch_arguments={
            "start_rqt": LaunchConfiguration("start_rqt"),
            "start_camera": LaunchConfiguration("start_camera"),
        }.items(),
    )

    qwen_selector = Node(
        package="idle_vision",
        executable="qwen_box_selector_node",
        name="qwen_box_selector_node",
        output="screen",
        parameters=[
            {
                "use_ollama": ParameterValue(
                    LaunchConfiguration("use_ollama"),
                    value_type=bool,
                ),
                "ollama_model": LaunchConfiguration("ollama_model"),
                "ollama_url": LaunchConfiguration("ollama_url"),
                "default_spatial": LaunchConfiguration("default_spatial"),
                "spatial_reference": LaunchConfiguration("spatial_reference"),
            }
        ],
    )

    voice_command = Node(
        package="idle_vision",
        executable="voice_command_node",
        name="voice_command_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_voice")),
        parameters=[
            {
                "whisper_model_size": LaunchConfiguration("whisper_model_size"),
                "whisper_device": LaunchConfiguration("whisper_device"),
                "whisper_compute_type": LaunchConfiguration("whisper_compute_type"),
                "input_device": LaunchConfiguration("input_device"),
                "energy_multiplier": ParameterValue(
                    LaunchConfiguration("energy_multiplier"),
                    value_type=float,
                ),
                "min_absolute_threshold": ParameterValue(
                    LaunchConfiguration("min_absolute_threshold"),
                    value_type=float,
                ),
                "log_energy": ParameterValue(
                    LaunchConfiguration("log_energy"),
                    value_type=bool,
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            start_rqt_arg,
            start_camera_arg,
            start_voice_arg,
            use_ollama_arg,
            ollama_model_arg,
            ollama_url_arg,
            default_spatial_arg,
            spatial_reference_arg,
            whisper_model_arg,
            whisper_device_arg,
            whisper_compute_arg,
            input_device_arg,
            energy_multiplier_arg,
            min_threshold_arg,
            log_energy_arg,
            box_pose_launch,
            qwen_selector,
            voice_command,
        ]
    )
