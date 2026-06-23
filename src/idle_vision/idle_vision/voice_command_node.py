"""Microphone VAD + Whisper STT bridge for Qwen box selection commands."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import wave
from collections import deque
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

try:
    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel
except ImportError as exc:  # pragma: no cover - depends on local audio packages.
    np = None
    sd = None
    WhisperModel = None
    _IMPORT_ERROR: Optional[ImportError] = exc
else:
    _IMPORT_ERROR = None


def _json_dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _normalize_exit_text(text: str) -> str:
    return (
        text.replace(" ", "")
        .replace(".", "")
        .replace(",", "")
        .replace("!", "")
        .replace("?", "")
        .strip()
    )


def _input_device_from_param(value: object) -> Optional[object]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


class VoiceCommandNode(Node):
    """Listen to the microphone, transcribe Korean speech, and publish commands."""

    def __init__(self) -> None:
        super().__init__("voice_command_node")

        self.declare_parameter("command_topic", "/idle_vision/qwen/command")
        self.declare_parameter("transcript_topic", "/idle_vision/voice/transcript")
        self.declare_parameter("status_topic", "/idle_vision/voice/status")

        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("frame_ms", 30)
        self.declare_parameter("input_device", "")
        self.declare_parameter("calibration_seconds", 1.0)
        self.declare_parameter("energy_multiplier", 1.4)
        self.declare_parameter("min_absolute_threshold", 800.0)
        self.declare_parameter("start_trigger_frames", 1)
        self.declare_parameter("end_silence_seconds", 1.0)
        self.declare_parameter("min_record_seconds", 0.5)
        self.declare_parameter("max_record_seconds", 8.0)
        self.declare_parameter("pre_roll_seconds", 0.3)
        self.declare_parameter("log_energy", False)
        self.declare_parameter("log_energy_every_frames", 10)

        self.declare_parameter("whisper_model_size", "small")
        self.declare_parameter("whisper_device", "cpu")
        self.declare_parameter("whisper_compute_type", "int8")
        self.declare_parameter("language", "ko")
        self.declare_parameter("beam_size", 5)
        self.declare_parameter("vad_filter", False)

        self.declare_parameter("print_audio_devices", False)
        self.declare_parameter("list_devices_only", False)
        self.declare_parameter("shutdown_on_exit_word", True)
        self.declare_parameter("exit_words", ["종료", "끝", "그만"])

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._transcript_topic = str(self.get_parameter("transcript_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)

        self._sample_rate = int(self.get_parameter("sample_rate").value)
        self._frame_ms = int(self.get_parameter("frame_ms").value)
        self._frame_size = int(self._sample_rate * self._frame_ms / 1000)
        self._input_device = _input_device_from_param(
            self.get_parameter("input_device").value
        )
        self._calibration_seconds = float(
            self.get_parameter("calibration_seconds").value
        )
        self._energy_multiplier = float(self.get_parameter("energy_multiplier").value)
        self._min_absolute_threshold = float(
            self.get_parameter("min_absolute_threshold").value
        )
        self._start_trigger_frames = int(
            self.get_parameter("start_trigger_frames").value
        )
        self._end_silence_seconds = float(
            self.get_parameter("end_silence_seconds").value
        )
        self._min_record_seconds = float(self.get_parameter("min_record_seconds").value)
        self._max_record_seconds = float(self.get_parameter("max_record_seconds").value)
        self._pre_roll_seconds = float(self.get_parameter("pre_roll_seconds").value)
        self._log_energy = bool(self.get_parameter("log_energy").value)
        self._log_energy_every_frames = max(
            int(self.get_parameter("log_energy_every_frames").value),
            1,
        )

        self._whisper_model_size = str(
            self.get_parameter("whisper_model_size").value
        )
        self._whisper_device = str(self.get_parameter("whisper_device").value)
        self._whisper_compute_type = str(
            self.get_parameter("whisper_compute_type").value
        )
        self._language = str(self.get_parameter("language").value)
        self._beam_size = int(self.get_parameter("beam_size").value)
        self._vad_filter = bool(self.get_parameter("vad_filter").value)

        self._print_audio_devices = bool(
            self.get_parameter("print_audio_devices").value
        )
        self._list_devices_only = bool(self.get_parameter("list_devices_only").value)
        self._shutdown_on_exit_word = bool(
            self.get_parameter("shutdown_on_exit_word").value
        )
        self._exit_words = {
            _normalize_exit_text(str(word))
            for word in self.get_parameter("exit_words").value
        }

        self._command_pub = self.create_publisher(String, self._command_topic, 10)
        self._transcript_pub = self.create_publisher(String, self._transcript_topic, 10)
        self._status_pub = self.create_publisher(String, self._status_topic, 10)

        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)

        if _IMPORT_ERROR is not None:
            message = (
                "voice_command_node requires faster_whisper, sounddevice, and numpy: "
                f"{_IMPORT_ERROR}"
            )
            self.get_logger().error(message)
            self._publish_status({"ok": False, "state": "dependency_error", "error": message})
            raise RuntimeError(message)

        self._worker.start()
        self.get_logger().info(
            "voice command ready: transcript=%s command=%s model=%s device=%s"
            % (
                self._transcript_topic,
                self._command_topic,
                self._whisper_model_size,
                self._whisper_device,
            )
        )

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        return super().destroy_node()

    def _run_loop(self) -> None:
        try:
            if self._print_audio_devices or self._list_devices_only:
                self._log_audio_devices()
            if self._list_devices_only:
                self._publish_status({"ok": True, "state": "listed_audio_devices"})
                rclpy.shutdown()
                return

            model = self._load_whisper_model()
            self._publish_status({"ok": True, "state": "listening"})

            while rclpy.ok() and not self._stop_event.is_set():
                audio_path = self._record_until_silence()
                if not audio_path:
                    continue

                try:
                    text = self._transcribe_audio(model, audio_path)
                finally:
                    try:
                        os.unlink(audio_path)
                    except OSError:
                        pass

                if not text:
                    self._publish_status({"ok": False, "state": "empty_transcript"})
                    continue

                self._publish_transcript(text)

                if _normalize_exit_text(text) in self._exit_words:
                    self._publish_status(
                        {"ok": True, "state": "exit_word", "transcript": text}
                    )
                    if self._shutdown_on_exit_word:
                        rclpy.shutdown()
                        return
                    continue

                self._publish_command(text)
                self._publish_status(
                    {"ok": True, "state": "command_published", "transcript": text}
                )

        except Exception as exc:  # pragma: no cover - depends on microphone runtime.
            self.get_logger().error(f"voice command loop stopped: {exc}")
            self._publish_status({"ok": False, "state": "error", "error": str(exc)})

    def _log_audio_devices(self) -> None:
        assert sd is not None
        devices = sd.query_devices()
        default_device = sd.default.device
        self.get_logger().info(f"audio default device: {default_device}")
        for index, device in enumerate(devices):
            max_inputs = int(device.get("max_input_channels", 0))
            max_outputs = int(device.get("max_output_channels", 0))
            name = str(device.get("name", "unknown"))
            self.get_logger().info(
                f"audio device {index}: {name} input={max_inputs} output={max_outputs}"
            )

    def _load_whisper_model(self):
        assert WhisperModel is not None
        self.get_logger().info("loading Whisper model...")
        self._publish_status(
            {
                "ok": True,
                "state": "loading_whisper",
                "model": self._whisper_model_size,
                "device": self._whisper_device,
                "compute_type": self._whisper_compute_type,
            }
        )
        model = WhisperModel(
            self._whisper_model_size,
            device=self._whisper_device,
            compute_type=self._whisper_compute_type,
        )
        self.get_logger().info("Whisper model loaded.")
        return model

    def _record_until_silence(self) -> Optional[str]:
        assert np is not None
        assert sd is not None

        pre_roll_maxlen = max(int(self._pre_roll_seconds * 1000 / self._frame_ms), 1)
        pre_roll_buffer = deque(maxlen=pre_roll_maxlen)
        recorded_frames = []
        speech_trigger_count = 0
        silence_seconds = 0.0
        recording = False
        recording_start_time = 0.0
        frame_counter = 0

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._frame_size,
            device=self._input_device,
        ) as stream:
            threshold = self._calibrate_noise(stream)
            self._publish_status(
                {"ok": True, "state": "listening", "vad_threshold": threshold}
            )

            while rclpy.ok() and not self._stop_event.is_set():
                frame, overflowed = stream.read(self._frame_size)
                energy = self._rms_energy(frame)
                frame_counter += 1

                if overflowed:
                    self._publish_status(
                        {"ok": False, "state": "audio_overflow", "energy": energy}
                    )

                if (
                    self._log_energy
                    and not recording
                    and frame_counter % self._log_energy_every_frames == 0
                ):
                    self.get_logger().info(
                        f"voice energy={energy:.1f} threshold={threshold:.1f}"
                    )

                if not recording:
                    pre_roll_buffer.append(frame.copy())
                    if energy > threshold:
                        speech_trigger_count += 1
                    else:
                        speech_trigger_count = 0

                    if speech_trigger_count >= self._start_trigger_frames:
                        recording = True
                        recording_start_time = time.time()
                        silence_seconds = 0.0
                        recorded_frames = list(pre_roll_buffer)
                        self._publish_status(
                            {"ok": True, "state": "recording", "energy": energy}
                        )
                else:
                    recorded_frames.append(frame.copy())
                    elapsed = time.time() - recording_start_time

                    if energy < threshold:
                        silence_seconds += self._frame_ms / 1000.0
                    else:
                        silence_seconds = 0.0

                    should_stop_by_silence = (
                        elapsed >= self._min_record_seconds
                        and silence_seconds >= self._end_silence_seconds
                    )
                    should_stop_by_max_time = elapsed >= self._max_record_seconds

                    if should_stop_by_silence or should_stop_by_max_time:
                        self._publish_status(
                            {
                                "ok": True,
                                "state": "recording_done",
                                "elapsed_s": elapsed,
                            }
                        )
                        break

        if self._stop_event.is_set() or not recorded_frames:
            return None

        audio = np.concatenate(recorded_frames, axis=0).reshape(-1)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        self._save_wav(temp_file.name, audio)
        return temp_file.name

    def _calibrate_noise(self, stream) -> float:
        energies = []
        num_frames = max(int(self._calibration_seconds * 1000 / self._frame_ms), 1)
        self._publish_status({"ok": True, "state": "calibrating_noise"})
        for _ in range(num_frames):
            if self._stop_event.is_set():
                break
            frame, _ = stream.read(self._frame_size)
            energies.append(self._rms_energy(frame))
        noise_level = float(np.median(energies)) if energies else 0.0
        threshold = max(
            noise_level * self._energy_multiplier,
            self._min_absolute_threshold,
        )
        self.get_logger().info(
            f"noise rms median={noise_level:.1f}, vad threshold={threshold:.1f}"
        )
        return threshold

    @staticmethod
    def _rms_energy(frame) -> float:
        frame_float = frame.astype(np.float32)
        return float(np.sqrt(np.mean(frame_float ** 2)))

    def _save_wav(self, path: str, audio) -> None:
        audio = np.asarray(audio, dtype=np.int16)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(audio.tobytes())

    def _transcribe_audio(self, model, audio_path: str) -> str:
        self._publish_status({"ok": True, "state": "transcribing"})
        segments, _ = model.transcribe(
            audio_path,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
        )
        text_parts = [segment.text.strip() for segment in segments]
        return " ".join(part for part in text_parts if part).strip()

    def _publish_transcript(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._transcript_pub.publish(msg)

    def _publish_command(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._command_pub.publish(msg)

    def _publish_status(self, payload: dict) -> None:
        msg = String()
        msg.data = _json_dumps(payload)
        self._status_pub.publish(msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = VoiceCommandNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
