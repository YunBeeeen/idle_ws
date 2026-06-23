"""Launch microphone VAD + Whisper STT command publisher."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    command_topic_arg = DeclareLaunchArgument(
        "command_topic",
        default_value="/idle_vision/qwen/command",
    )
    transcript_topic_arg = DeclareLaunchArgument(
        "transcript_topic",
        default_value="/idle_vision/voice/transcript",
    )
    status_topic_arg = DeclareLaunchArgument(
        "status_topic",
        default_value="/idle_vision/voice/status",
    )
    input_device_arg = DeclareLaunchArgument("input_device", default_value="")
    whisper_model_arg = DeclareLaunchArgument("whisper_model_size", default_value="small")
    whisper_device_arg = DeclareLaunchArgument("whisper_device", default_value="cpu")
    whisper_compute_arg = DeclareLaunchArgument("whisper_compute_type", default_value="int8")
    energy_multiplier_arg = DeclareLaunchArgument("energy_multiplier", default_value="1.4")
    min_threshold_arg = DeclareLaunchArgument("min_absolute_threshold", default_value="800.0")
    print_devices_arg = DeclareLaunchArgument("print_audio_devices", default_value="false")
    list_devices_only_arg = DeclareLaunchArgument("list_devices_only", default_value="false")
    log_energy_arg = DeclareLaunchArgument("log_energy", default_value="false")

    voice_command = Node(
        package="idle_vision",
        executable="voice_command_node",
        name="voice_command_node",
        output="screen",
        parameters=[
            {
                "command_topic": LaunchConfiguration("command_topic"),
                "transcript_topic": LaunchConfiguration("transcript_topic"),
                "status_topic": LaunchConfiguration("status_topic"),
                "input_device": LaunchConfiguration("input_device"),
                "whisper_model_size": LaunchConfiguration("whisper_model_size"),
                "whisper_device": LaunchConfiguration("whisper_device"),
                "whisper_compute_type": LaunchConfiguration("whisper_compute_type"),
                "energy_multiplier": ParameterValue(
                    LaunchConfiguration("energy_multiplier"),
                    value_type=float,
                ),
                "min_absolute_threshold": ParameterValue(
                    LaunchConfiguration("min_absolute_threshold"),
                    value_type=float,
                ),
                "print_audio_devices": ParameterValue(
                    LaunchConfiguration("print_audio_devices"),
                    value_type=bool,
                ),
                "list_devices_only": ParameterValue(
                    LaunchConfiguration("list_devices_only"),
                    value_type=bool,
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
            command_topic_arg,
            transcript_topic_arg,
            status_topic_arg,
            input_device_arg,
            whisper_model_arg,
            whisper_device_arg,
            whisper_compute_arg,
            energy_multiplier_arg,
            min_threshold_arg,
            print_devices_arg,
            list_devices_only_arg,
            log_energy_arg,
            voice_command,
        ]
    )
