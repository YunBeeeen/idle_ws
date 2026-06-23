"""Slow joint-space sweep controller for dynamic no-contact logging.

This node is intended for contact-detector validation on the real robot.
It publishes gravity-compensated MIT commands while moving selected joints
through a sign-safe default sequence:

    current -> 0 -> -amplitude -> 0 -> +amplitude -> 0

The zero waypoints are explicit so joints such as j2/j3 never jump directly
from the positive side to the negative side.

For real-robot bring-up, a custom one-joint sweep is also available:

    current -> custom_high -> custom_low -> custom_high

with an optional fixed motor target, e.g. j2 held near a home pose while only
j3 moves.

For collecting real no-contact hard negatives, free-motion waypoint mode can
play a slow sequence of joint-space waypoints without editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from xml.etree import ElementTree as ET

import rclpy
from idle_common.control_tuning import control_params_for_motor
from idle_common.motor_map import (
    DEFAULT_MOTOR_JOINT_MAP,
    DEFAULT_TAU_LIMIT_BY_MOTOR,
    parse_float_map_json,
    parse_motor_joint_map_json,
)
from idle_common.paths import resolve_share_file
from idle_common.ros_params import declare_typed
from msgs.msg import MotorCMD, MotorCMDArray, MotorStateArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from phy.collision import CollisionChecker
from phy.gravity import GravityCompensator
from phy.robot_model import RobotModel


@dataclass
class MotorSample:
    q: float = 0.0
    qd: float = 0.0
    tau_measured: float = 0.0
    last_seen_s: float = float("-inf")


@dataclass(frozen=True)
class Segment:
    start: dict[int, float]
    end: dict[int, float]
    duration_s: float
    label: str


def _clip_symmetric(value: float, limit: float) -> float:
    if not math.isfinite(limit) or limit <= 0.0:
        return float(value)
    return float(max(-limit, min(limit, value)))


def _as_float_or_default(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _parse_int_list_value(value: object, field_name: str) -> list[int]:
    if isinstance(value, str):
        text_stripped = value.strip()
        if not text_stripped:
            return []
        try:
            raw = json.loads(text_stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be valid JSON list: {exc}") from exc
    else:
        raw = value
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a JSON list or ROS integer array")
    out: list[int] = []
    for value in raw:
        try:
            motor_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}: invalid motor id '{value}'") from exc
        out.append(motor_id)
    return out


def _smoothstep5(u: float) -> tuple[float, float]:
    """Return quintic smoothstep position and derivative with respect to u."""

    u_clamped = max(0.0, min(1.0, float(u)))
    pos = 10.0 * u_clamped**3 - 15.0 * u_clamped**4 + 6.0 * u_clamped**5
    vel_du = 30.0 * u_clamped**2 - 60.0 * u_clamped**3 + 30.0 * u_clamped**4
    return pos, vel_du


def _load_urdf_joint_limits(
    urdf_path: str | Path,
    motor_joint_map: dict[int, str],
    fallback_abs_limit_rad: float,
) -> dict[int, tuple[float, float]]:
    """Load motor-id keyed position limits from URDF, with a safe fallback."""

    root = ET.parse(Path(urdf_path).expanduser().resolve()).getroot()
    joint_limits_by_name: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        joint_name = str(joint.attrib.get("name", "")).strip()
        if not joint_name:
            continue
        joint_type = str(joint.attrib.get("type", "")).strip()
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            joint_limits_by_name[joint_name] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
        elif joint_type == "continuous":
            joint_limits_by_name[joint_name] = (-fallback_abs_limit_rad, fallback_abs_limit_rad)

    out: dict[int, tuple[float, float]] = {}
    for motor_id, joint_name in motor_joint_map.items():
        out[int(motor_id)] = joint_limits_by_name.get(
            joint_name,
            (-fallback_abs_limit_rad, fallback_abs_limit_rad),
        )
    return out


def _sign_crosses_zero(a: float, b: float, eps: float) -> bool:
    return abs(a) > eps and abs(b) > eps and ((a > 0.0 and b < 0.0) or (a < 0.0 and b > 0.0))


class JointSweepNode(Node):
    """Publish a slow j2/j3 sweep with gravity compensation."""

    def __init__(self) -> None:
        super().__init__("joint_sweep_node")

        self.control_hz = declare_typed(self, "control_hz", 250.0)
        self.state_timeout_s = declare_typed(self, "state_timeout_s", 0.2)
        self.stale_warn_throttle_s = declare_typed(self, "stale_warn_throttle_s", 2.0)
        self.state_max_abs_q_rad = declare_typed(self, "state_max_abs_q_rad", math.pi + 0.25)
        self.state_max_abs_qd_rad_s = declare_typed(self, "state_max_abs_qd_rad_s", 80.0)
        self.state_max_abs_tau_nm = declare_typed(self, "state_max_abs_tau_nm", 120.0)
        self.start_state_joint_limit_tolerance_rad = max(
            0.0, declare_typed(self, "start_state_joint_limit_tolerance_rad", 0.03)
        )
        self.amplitude_rad = abs(declare_typed(self, "amplitude_rad", 0.8))
        self.segment_duration_s = max(0.5, declare_typed(self, "segment_duration_s", 10.0))
        self.initial_to_zero_duration_s = max(
            0.5, declare_typed(self, "initial_to_zero_duration_s", 6.0)
        )
        self.start_all_motors_at_zero = declare_typed(self, "start_all_motors_at_zero", True)
        self.zero_rad = declare_typed(self, "zero_rad", 0.0)
        self.repeat = declare_typed(self, "repeat", False)
        self.sweep_kp = declare_typed(self, "sweep_kp", 3.0)
        self.sweep_kd = declare_typed(self, "sweep_kd", 0.25)
        self.hold_kp = declare_typed(self, "hold_kp", 1.5)
        self.hold_kd = declare_typed(self, "hold_kd", 0.10)
        self.use_tuning_gains = declare_typed(self, "use_tuning_gains", False)
        self.custom_sweep_motor_id = declare_typed(self, "custom_sweep_motor_id", 0)
        self.custom_sweep_high_rad = declare_typed(self, "custom_sweep_high_rad", -0.4)
        self.custom_sweep_low_rad = declare_typed(self, "custom_sweep_low_rad", -1.5)
        self.fixed_motor_id = declare_typed(self, "fixed_motor_id", 0)
        self.fixed_motor_q_rad = declare_typed(self, "fixed_motor_q_rad", 1.0)
        self.free_motion_preset = declare_typed(
            self, "free_motion_preset", "", cast=lambda value: str(value).strip()
        )
        self.free_motion_fixed_j2_rad = declare_typed(self, "free_motion_fixed_j2_rad", -0.825)
        self.random_motion_enabled = declare_typed(self, "random_motion_enabled", False)
        self.random_waypoint_count = max(1, declare_typed(self, "random_waypoint_count", 10))
        self.random_seed = declare_typed(self, "random_seed", 7)
        self.random_max_attempts = max(1, declare_typed(self, "random_max_attempts", 500))
        self.sine_motion_enabled = declare_typed(self, "sine_motion_enabled", False)
        self.sine_envelope_enabled = declare_typed(self, "sine_envelope_enabled", True)
        self.sine_envelope_period_s = max(1.0, declare_typed(self, "sine_envelope_period_s", 80.0))
        self.sine_envelope_min_scale = max(
            0.0, min(1.0, declare_typed(self, "sine_envelope_min_scale", 0.35))
        )
        self.sine_validation_duration_s = max(
            1.0, declare_typed(self, "sine_validation_duration_s", 120.0)
        )
        self.sine_duration_s = max(0.0, declare_typed(self, "sine_duration_s", 0.0))
        self.enforce_zero_crossing = declare_typed(self, "enforce_zero_crossing", True)
        self.zero_crossing_epsilon_rad = declare_typed(self, "zero_crossing_epsilon_rad", 1.0e-4)
        self.enforce_joint_limits = declare_typed(self, "enforce_joint_limits", True)
        self.max_abs_joint_rad = declare_typed(self, "max_abs_joint_rad", math.pi)
        self.check_self_collision = declare_typed(self, "check_self_collision", True)
        self.collision_samples_per_segment = max(
            2, declare_typed(self, "collision_samples_per_segment", 25)
        )
        self.check_floor_clearance = declare_typed(self, "check_floor_clearance", False)
        self.min_floor_clearance_m = declare_typed(self, "min_floor_clearance_m", 0.02)
        self.check_gravity_load = declare_typed(self, "check_gravity_load", True)
        self.random_min_step_norm_rad = declare_typed(self, "random_min_step_norm_rad", 0.25)
        self.random_max_step_norm_rad = declare_typed(self, "random_max_step_norm_rad", 1.10)

        strip_str = lambda v: str(v).strip()
        default_map_json = json.dumps(DEFAULT_MOTOR_JOINT_MAP)
        default_limit_json = json.dumps(DEFAULT_TAU_LIMIT_BY_MOTOR)
        map_json_text = declare_typed(self, "motor_joint_map_json", default_map_json)
        limit_json_text = declare_typed(self, "tau_limit_by_motor_json", default_limit_json)
        sweep_ids_value = declare_typed(
            self, "sweep_motor_ids_json", [2, 3], cast=lambda value: value
        )
        random_ids_value = declare_typed(
            self, "random_motor_ids_json", [2, 3], cast=lambda value: value
        )
        sine_ids_value = declare_typed(
            self, "sine_motor_ids_json", [2, 3], cast=lambda value: value
        )
        sine_center_text = declare_typed(
            self,
            "sine_center_by_motor_json",
            '{"2": -0.70, "3": -1.30}',
            cast=lambda value: str(value).strip(),
        )
        sine_amplitude_text = declare_typed(
            self,
            "sine_amplitude_by_motor_json",
            '{"2": 0.08, "3": 0.08}',
            cast=lambda value: str(value).strip(),
        )
        sine_frequency_text = declare_typed(
            self,
            "sine_frequency_by_motor_json",
            '{"2": 0.025, "3": 0.040}',
            cast=lambda value: str(value).strip(),
        )
        sine_phase_text = declare_typed(
            self,
            "sine_phase_by_motor_json",
            '{"2": 0.0, "3": 1.57079632679}',
            cast=lambda value: str(value).strip(),
        )
        random_range_text = declare_typed(
            self,
            "random_range_by_motor_json",
            '{"2": [-1.20, -0.45], "3": [-1.55, -0.45]}',
            cast=lambda value: str(value).strip(),
        )
        gravity_load_limit_text = declare_typed(
            self,
            "gravity_load_limit_by_motor_json",
            '{"2": 25.0, "3": 11.0}',
            cast=lambda value: str(value).strip(),
        )
        waypoints_path_text = declare_typed(self, "waypoints_path", "", cast=strip_str)
        urdf_path_text = declare_typed(self, "urdf_path", "", cast=strip_str)
        srdf_path_text = declare_typed(self, "srdf_path", "", cast=strip_str)
        model_xml_text = declare_typed(self, "model_xml", "", cast=strip_str)
        floor_exclude_body_names_text = declare_typed(
            self,
            "floor_exclude_body_names_json",
            '["world", "joint1"]',
            cast=strip_str,
        )

        motor_joint_map = parse_motor_joint_map_json(map_json_text)
        if not motor_joint_map:
            motor_joint_map = dict(DEFAULT_MOTOR_JOINT_MAP)
        tau_limit_map = parse_float_map_json(limit_json_text, "tau_limit_by_motor_json")
        if not tau_limit_map:
            tau_limit_map = dict(DEFAULT_TAU_LIMIT_BY_MOTOR)
        sweep_motor_ids = _parse_int_list_value(sweep_ids_value, "sweep_motor_ids_json")
        if int(self.custom_sweep_motor_id) > 0:
            sweep_motor_ids = [int(self.custom_sweep_motor_id)]
        elif not sweep_motor_ids:
            sweep_motor_ids = [2, 3]

        urdf_path = resolve_share_file("sim", "urdf/robot.urdf", urdf_path_text)
        srdf_path = resolve_share_file("sim", "srdf/robot.srdf", srdf_path_text)
        model_xml_path = resolve_share_file("sim", "robot.xml", model_xml_text)
        self.gravity = GravityCompensator(urdf_path, motor_joint_map)
        self.joint_limits_by_motor = _load_urdf_joint_limits(
            urdf_path,
            motor_joint_map,
            float(self.max_abs_joint_rad),
        )
        self.collision_checker: CollisionChecker | None = None
        if self.check_self_collision:
            from ament_index_python.packages import get_package_share_directory

            sim_share_parent = str(Path(get_package_share_directory("sim")).parent)
            robot_model = RobotModel(urdf_path, motor_joint_map)
            self.collision_checker = CollisionChecker(
                robot_model,
                srdf_path=srdf_path,
                package_dirs=[sim_share_parent],
            )
        self.floor_model = None
        self.floor_data = None
        self.floor_qpos_idx_by_motor: dict[int, int] = {}
        self.floor_checked_geom_ids: list[int] = []
        self.floor_mujoco = None
        if self.check_floor_clearance:
            self._warn_if_model_xml_stale(urdf_path, model_xml_path)
            self._init_floor_clearance_checker(model_xml_path, motor_joint_map, floor_exclude_body_names_text)

        self.motor_joint_map = motor_joint_map
        self.motor_ids = self.gravity.ordered_motor_ids
        self.fixed_targets_by_motor: dict[int, float] = {}
        if int(self.fixed_motor_id) > 0:
            fixed_motor_id = int(self.fixed_motor_id)
            if fixed_motor_id in self.motor_ids:
                self.fixed_targets_by_motor[fixed_motor_id] = float(self.fixed_motor_q_rad)
            else:
                self.get_logger().warn(f"ignoring fixed_motor_id not in URDF motor map: {fixed_motor_id}")

        self.random_motor_ids = [
            motor_id
            for motor_id in _parse_int_list_value(random_ids_value, "random_motor_ids_json")
            if motor_id in self.motor_ids
        ]
        if not self.random_motor_ids:
            self.random_motor_ids = [motor_id for motor_id in (2, 3) if motor_id in self.motor_ids]
        self.random_ranges_by_motor = self._parse_random_ranges(random_range_text)
        self.gravity_load_limit_by_motor = parse_float_map_json(
            gravity_load_limit_text,
            "gravity_load_limit_by_motor_json",
        )
        self.sine_motor_ids = [
            motor_id
            for motor_id in _parse_int_list_value(sine_ids_value, "sine_motor_ids_json")
            if motor_id in self.motor_ids
        ]
        if not self.sine_motor_ids:
            self.sine_motor_ids = [motor_id for motor_id in (2, 3) if motor_id in self.motor_ids]
        self.sine_center_by_motor = parse_float_map_json(
            sine_center_text,
            "sine_center_by_motor_json",
        )
        self.sine_amplitude_by_motor = parse_float_map_json(
            sine_amplitude_text,
            "sine_amplitude_by_motor_json",
        )
        self.sine_frequency_by_motor = parse_float_map_json(
            sine_frequency_text,
            "sine_frequency_by_motor_json",
        )
        self.sine_phase_by_motor = parse_float_map_json(
            sine_phase_text,
            "sine_phase_by_motor_json",
        )

        self.waypoint_targets = self._load_waypoint_targets(waypoints_path_text)
        if self.sine_motion_enabled:
            self.waypoint_targets = []
            sweep_motor_ids = list(self.sine_motor_ids)
        elif not self.waypoint_targets and self.random_motion_enabled:
            self.waypoint_targets = self._generate_random_waypoint_targets()
        if not self.sine_motion_enabled and not self.waypoint_targets:
            self.waypoint_targets = self._preset_waypoint_targets(self.free_motion_preset)
        if not self.sine_motion_enabled and self.waypoint_targets:
            sweep_motor_ids = sorted({motor_id for waypoint in self.waypoint_targets for motor_id in waypoint})

        self.sweep_motor_ids = [motor_id for motor_id in sweep_motor_ids if motor_id in self.motor_ids]
        if not self.sweep_motor_ids:
            raise ValueError("sweep_motor_ids_json/custom/waypoints did not contain any motor from the URDF motor map")
        ignored_ids = sorted(set(sweep_motor_ids) - set(self.sweep_motor_ids))
        if ignored_ids:
            self.get_logger().warn(f"ignoring sweep motors not in URDF motor map: {ignored_ids}")
        self.required_motor_ids = sorted(set(self.sweep_motor_ids) | set(self.fixed_targets_by_motor.keys()))

        self.tau_limit_by_motor = {
            motor_id: float(tau_limit_map.get(motor_id, float("inf"))) for motor_id in self.motor_ids
        }
        self.state_by_motor = {motor_id: MotorSample() for motor_id in self.motor_ids}
        self.last_connected_ids: tuple[int, ...] = tuple()
        self.last_stale_warn_s = float("-inf")
        self.last_bad_state_warn_s = float("-inf")

        self.initial_q_by_motor: dict[int, float] | None = None
        self.motion_start_q_by_motor: dict[int, float] | None = None
        self.sine_reference_q_by_motor: dict[int, float] | None = None
        self.segments: list[Segment] = []
        self.active_segment_idx = 0
        self.segment_started_s = float("nan")
        self.sine_started_s = float("nan")
        self.sine_warmup_complete = False
        self.sequence_complete = False
        self.last_segment_log_idx = -1

        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_state = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.state_sub = self.create_subscription(
            MotorStateArray, "/motor_state_array", self.on_state_array, qos_state
        )
        self.cmd_pub = self.create_publisher(MotorCMDArray, "/motor_cmd_array", qos_cmd)

        period_s = max(1.0 / self.control_hz, 1.0e-4)
        self.control_timer = self.create_timer(period_s, self.on_timer)

        self.get_logger().info(
            "joint_sweep_node initialized: "
            f"sweep_motors={self.sweep_motor_ids} "
            f"custom_sweep_motor_id={int(self.custom_sweep_motor_id)} "
            f"default_sequence=0 -> -{self.amplitude_rad:.3f} -> 0 -> +{self.amplitude_rad:.3f} -> 0 "
            f"custom_sequence={self.custom_sweep_high_rad:.3f} -> {self.custom_sweep_low_rad:.3f} "
            f"-> {self.custom_sweep_high_rad:.3f} "
            f"fixed_targets={self.fixed_targets_by_motor} "
            f"free_motion_preset={self.free_motion_preset or 'disabled'} "
            f"random_motion_enabled={self.random_motion_enabled} "
            f"sine_motion_enabled={self.sine_motion_enabled} "
            f"random_seed={self.random_seed} "
            f"waypoint_count={len(self.waypoint_targets)} "
            f"check_floor_clearance={self.check_floor_clearance} "
            f"check_gravity_load={self.check_gravity_load} "
            f"segment_duration={self.segment_duration_s:.2f}s "
            f"initial_to_zero={self.initial_to_zero_duration_s:.2f}s "
            f"start_all_motors_at_zero={self.start_all_motors_at_zero} "
            f"start_state_joint_limit_tolerance={self.start_state_joint_limit_tolerance_rad:.3f}rad "
            f"sweep_kp/kd={self.sweep_kp:.3f}/{self.sweep_kd:.3f} "
            f"hold_kp/kd={self.hold_kp:.3f}/{self.hold_kd:.3f} "
            f"repeat={self.repeat} urdf={urdf_path}"
        )

    def on_state_array(self, msg: MotorStateArray) -> None:
        now_s = time.monotonic()
        bad_states: list[str] = []
        for state in msg.states:
            motor_id = int(state.motor_id)
            sample = self.state_by_motor.get(motor_id)
            if sample is None:
                continue
            q = float(state.q)
            qd = float(state.qd)
            tau = float(state.tau)
            is_bad = False
            if not math.isfinite(q) or not math.isfinite(qd) or not math.isfinite(tau):
                bad_states.append(f"m{motor_id}: non-finite")
                is_bad = True
            elif abs(q) > float(self.state_max_abs_q_rad):
                bad_states.append(f"m{motor_id}: |q|={abs(q):.3f}")
                is_bad = True
            elif abs(qd) > float(self.state_max_abs_qd_rad_s):
                bad_states.append(f"m{motor_id}: |qd|={abs(qd):.3f}")
                is_bad = True
            elif abs(tau) > float(self.state_max_abs_tau_nm):
                bad_states.append(f"m{motor_id}: |tau|={abs(tau):.3f}")
                is_bad = True
            if is_bad:
                continue
            sample.q = float(state.q)
            sample.qd = float(state.qd)
            sample.tau_measured = float(state.tau)
            sample.last_seen_s = now_s
        if bad_states and (now_s - self.last_bad_state_warn_s) >= self.stale_warn_throttle_s:
            self.get_logger().warn(
                "discarding implausible motor state fields: " + ", ".join(bad_states[:6])
            )
            self.last_bad_state_warn_s = now_s

    def _connected_motor_ids(self, now_s: float) -> list[int]:
        connected: list[int] = []
        for motor_id in self.motor_ids:
            sample = self.state_by_motor[motor_id]
            if not math.isfinite(sample.last_seen_s):
                continue
            if (now_s - sample.last_seen_s) <= self.state_timeout_s:
                connected.append(motor_id)
        return connected

    def _all_zero_q_by_motor(self) -> dict[int, float]:
        return {motor_id: 0.0 for motor_id in self.motor_ids}

    def _build_all_zero_start_segments(self, initial_q: dict[int, float]) -> list[Segment]:
        if not self.start_all_motors_at_zero:
            return []

        zero_q = self._all_zero_q_by_motor()
        if all(abs(float(initial_q.get(motor_id, 0.0))) <= 1.0e-4 for motor_id in self.motor_ids):
            return []

        return [
            Segment(
                start=dict(initial_q),
                end=zero_q,
                duration_s=self.initial_to_zero_duration_s,
                label="current_to_all_zero",
            )
        ]

    def _ensure_sequence_initialized(self, now_s: float, connected_ids: list[int]) -> bool:
        if self.initial_q_by_motor is not None:
            return True
        missing = [motor_id for motor_id in self.required_motor_ids if motor_id not in connected_ids]
        if missing:
            self.get_logger().warn(
                f"waiting for required motor states before starting sequence: missing={missing}"
            )
            return False
        self.initial_q_by_motor = {
            motor_id: self.state_by_motor[motor_id].q for motor_id in self.motor_ids
        }
        self.motion_start_q_by_motor = (
            self._all_zero_q_by_motor()
            if self.start_all_motors_at_zero
            else dict(self.initial_q_by_motor)
        )
        zero_start_segments = self._build_all_zero_start_segments(self.initial_q_by_motor)

        if self.sine_motion_enabled:
            self.sine_reference_q_by_motor = dict(self.motion_start_q_by_motor)
            sine_start_q, _sine_start_qd = self._sample_sine_at_elapsed(0.0)
            sine_warmup_segments = list(zero_start_segments)
            if any(
                abs(float(self.motion_start_q_by_motor[motor_id]) - float(sine_start_q[motor_id])) > 1.0e-4
                for motor_id in self.motor_ids
            ):
                sine_warmup_segments.append(
                    Segment(
                        start=dict(self.motion_start_q_by_motor),
                        end=sine_start_q,
                        duration_s=self.initial_to_zero_duration_s,
                        label="zero_to_sine_start" if self.start_all_motors_at_zero else "sine_warmup",
                    )
                )

            self.segments = self._insert_zero_crossing_segments(sine_warmup_segments)
            if self.segments:
                self._validate_segments_or_raise(self.segments)
            self._validate_sine_motion_or_raise()
            self.active_segment_idx = 0
            self.segment_started_s = now_s
            self.sine_warmup_complete = not bool(self.segments)
            self.sine_started_s = now_s if self.sine_warmup_complete else float("nan")
            self.sequence_complete = False
            self.last_segment_log_idx = -1
            self.get_logger().info(
                "sine motion initialized from "
                + ", ".join(
                    f"m{motor_id}={self.initial_q_by_motor[motor_id]:+.3f}rad"
                    for motor_id in self.required_motor_ids
                )
                + " via all-zero start to "
                + ", ".join(
                    f"m{motor_id}={sine_start_q[motor_id]:+.3f}rad"
                    for motor_id in self.required_motor_ids
                )
                + f" centers={self.sine_center_by_motor} "
                + f"amplitudes={self.sine_amplitude_by_motor} "
                + f"frequencies_hz={self.sine_frequency_by_motor}"
            )
            return True
        self.segments = zero_start_segments + self._build_segments(self.motion_start_q_by_motor)
        self.segments = self._insert_zero_crossing_segments(self.segments)
        self._validate_segments_or_raise(self.segments)
        self.active_segment_idx = 0
        self.segment_started_s = now_s
        self.sequence_complete = False
        self.last_segment_log_idx = -1
        self.get_logger().info(
            "sweep sequence started from "
            + ", ".join(
                f"m{motor_id}={self.initial_q_by_motor[motor_id]:+.3f}rad"
                for motor_id in self.required_motor_ids
            )
            + (
                " with all-motor zero-start enabled"
                if self.start_all_motors_at_zero
                else ""
            )
        )
        return True

    def _insert_zero_crossing_segments(self, segments: list[Segment]) -> list[Segment]:
        if not self.enforce_zero_crossing:
            return segments

        out: list[Segment] = []
        eps = float(self.zero_crossing_epsilon_rad)
        for segment in segments:
            crossing_motors: list[int] = []
            motor_ids = sorted(set(segment.start.keys()) | set(segment.end.keys()))
            for motor_id in motor_ids:
                start_q = float(segment.start.get(motor_id, 0.0))
                end_q = float(segment.end.get(motor_id, start_q))
                if _sign_crosses_zero(start_q, end_q, eps):
                    crossing_motors.append(motor_id)

            if not crossing_motors:
                out.append(segment)
                continue

            midpoint = dict(segment.start)
            for motor_id in motor_ids:
                start_q = float(segment.start.get(motor_id, 0.0))
                end_q = float(segment.end.get(motor_id, start_q))
                midpoint[motor_id] = 0.0 if motor_id in crossing_motors else 0.5 * (start_q + end_q)

            half_duration_s = max(0.5 * segment.duration_s, 0.5)
            self.get_logger().warn(
                f"segment '{segment.label}' crosses sign for motors {crossing_motors}; "
                "inserting explicit zero-crossing waypoint"
            )
            out.append(
                Segment(
                    start=segment.start,
                    end=midpoint,
                    duration_s=half_duration_s,
                    label=f"{segment.label}_to_zero",
                )
            )
            out.append(
                Segment(
                    start=midpoint,
                    end=segment.end,
                    duration_s=half_duration_s,
                    label=f"{segment.label}_from_zero",
                )
            )
        return out

    def _validate_target_or_raise(self, motor_id: int, q_rad: float, label: str) -> None:
        if not math.isfinite(q_rad):
            raise ValueError(f"{label}: motor {motor_id} target is not finite: {q_rad}")
        max_abs = float(self.max_abs_joint_rad)
        eps = 1.0e-6
        if abs(q_rad) > max_abs + eps:
            raise ValueError(
                f"{label}: motor {motor_id} target {q_rad:.6f} rad exceeds "
                f"absolute safety limit +/-{max_abs:.6f} rad"
            )
        if not self.enforce_joint_limits:
            return
        lower, upper = self.joint_limits_by_motor.get(motor_id, (-max_abs, max_abs))
        if q_rad < lower - eps or q_rad > upper + eps:
            joint_name = self.motor_joint_map.get(motor_id, f"motor_{motor_id}")
            raise ValueError(
                f"{label}: motor {motor_id} ({joint_name}) target {q_rad:.6f} rad "
                f"outside URDF joint limit [{lower:.6f}, {upper:.6f}]"
            )

    def _validate_segment_endpoint_or_raise(
        self,
        motor_id: int,
        q_rad: float,
        label: str,
        *,
        is_start_endpoint: bool,
    ) -> None:
        if not is_start_endpoint:
            self._validate_target_or_raise(motor_id, q_rad, label)
            return

        if not math.isfinite(q_rad):
            raise ValueError(f"{label}: motor {motor_id} start state is not finite: {q_rad}")
        max_abs = float(self.max_abs_joint_rad)
        start_tol = float(self.start_state_joint_limit_tolerance_rad)
        if abs(q_rad) > max_abs + start_tol:
            raise ValueError(
                f"{label}: motor {motor_id} start state {q_rad:.6f} rad exceeds "
                f"absolute safety limit +/-{max_abs:.6f} rad with start tolerance {start_tol:.6f} rad"
            )
        if not self.enforce_joint_limits:
            return

        lower, upper = self.joint_limits_by_motor.get(motor_id, (-max_abs, max_abs))
        if q_rad < lower - start_tol or q_rad > upper + start_tol:
            joint_name = self.motor_joint_map.get(motor_id, f"motor_{motor_id}")
            raise ValueError(
                f"{label}: motor {motor_id} ({joint_name}) start state {q_rad:.6f} rad "
                f"outside URDF joint limit [{lower:.6f}, {upper:.6f}] even with "
                f"start tolerance {start_tol:.6f} rad"
            )

    def _validate_segments_or_raise(self, segments: list[Segment]) -> None:
        if not segments:
            raise ValueError("motion sequence has no valid segments")

        for segment in segments:
            segment_motor_ids = sorted(
                (set(segment.start.keys()) | set(segment.end.keys()) | set(self.required_motor_ids))
                & set(self.motor_ids)
            )
            for endpoint_name, target in (("start", segment.start), ("end", segment.end)):
                for motor_id in segment_motor_ids:
                    self._validate_segment_endpoint_or_raise(
                        motor_id,
                        float(target.get(motor_id, 0.0)),
                        f"{segment.label}.{endpoint_name}",
                        is_start_endpoint=(endpoint_name == "start"),
                    )
            if self.enforce_zero_crossing:
                for motor_id in segment_motor_ids:
                    start_q = float(segment.start.get(motor_id, 0.0))
                    end_q = float(segment.end.get(motor_id, start_q))
                    if _sign_crosses_zero(start_q, end_q, float(self.zero_crossing_epsilon_rad)):
                        raise ValueError(
                            f"{segment.label}: motor {motor_id} still crosses sign without zero waypoint "
                            f"({start_q:.6f} -> {end_q:.6f})"
                        )

        self._validate_collision_or_raise(segments)
        self._validate_floor_clearance_or_raise(segments)
        self._validate_gravity_load_or_raise(segments)

    def _validate_collision_or_raise(self, segments: list[Segment]) -> None:
        if self.collision_checker is None:
            return

        sample_count = max(2, int(self.collision_samples_per_segment))
        for segment in segments:
            samples: list[dict[int, float]] = []
            for idx in range(sample_count):
                u = idx / max(sample_count - 1, 1)
                alpha, _ = _smoothstep5(u)
                q_sample: dict[int, float] = {}
                for motor_id in self.motor_ids:
                    start_q = float(segment.start.get(motor_id, 0.0))
                    end_q = float(segment.end.get(motor_id, start_q))
                    q_sample[motor_id] = start_q + (end_q - start_q) * alpha
                samples.append(q_sample)

            any_collision, first_idx = self.collision_checker.check_trajectory(samples)
            if not any_collision:
                continue

            colliding_pairs = self.collision_checker.colliding_pairs(samples[first_idx])
            raise ValueError(
                f"{segment.label}: self-collision detected at sampled index "
                f"{first_idx}/{sample_count - 1}; pairs={colliding_pairs[:5]}"
            )

    def _warn_if_model_xml_stale(self, urdf_path: str | Path, model_xml_path: str | Path) -> None:
        urdf = Path(urdf_path).expanduser().resolve()
        model_xml = Path(model_xml_path).expanduser().resolve()
        if not urdf.exists() or not model_xml.exists():
            return
        if urdf.stat().st_mtime > model_xml.stat().st_mtime + 1.0:
            self.get_logger().warn(
                "URDF is newer than MuJoCo robot.xml. Regenerate sim/robot.xml before trusting "
                f"floor-clearance preview: urdf={urdf} model_xml={model_xml}"
            )

    def _parse_floor_excluded_body_names(self, text: str) -> set[str]:
        try:
            raw = json.loads(text) if text else []
        except json.JSONDecodeError as exc:
            raise ValueError(f"floor_exclude_body_names_json must be valid JSON list: {exc}") from exc
        if not isinstance(raw, list):
            raise ValueError("floor_exclude_body_names_json must be a JSON list")
        return {str(item).strip() for item in raw if str(item).strip()}

    def _init_floor_clearance_checker(
        self,
        model_xml_path: str | Path,
        motor_joint_map: dict[int, str],
        floor_exclude_body_names_text: str,
    ) -> None:
        import mujoco
        from sim.viewer_node import load_model_with_workaround

        model, _used_workaround = load_model_with_workaround(str(model_xml_path))
        data = mujoco.MjData(model)
        qpos_idx_by_motor: dict[int, int] = {}
        for motor_id, joint_name in sorted(motor_joint_map.items()):
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            if joint_id < 0:
                raise ValueError(f"joint '{joint_name}' not in MuJoCo model (motor_id={motor_id})")
            qpos_idx_by_motor[int(motor_id)] = int(model.jnt_qposadr[joint_id])

        excluded_body_names = self._parse_floor_excluded_body_names(floor_exclude_body_names_text)
        checked_geom_ids: list[int] = []
        for geom_id in range(int(model.ngeom)):
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            body_id = int(model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if body_name in excluded_body_names:
                continue
            checked_geom_ids.append(geom_id)

        if not checked_geom_ids:
            raise ValueError("floor clearance checker has no mesh geoms to check")

        self.floor_mujoco = mujoco
        self.floor_model = model
        self.floor_data = data
        self.floor_qpos_idx_by_motor = qpos_idx_by_motor
        self.floor_checked_geom_ids = checked_geom_ids
        self.get_logger().info(
            f"floor clearance checker initialized: checked_geoms={len(checked_geom_ids)} "
            f"excluded_bodies={sorted(excluded_body_names)} min_z={self.min_floor_clearance_m:.3f}m"
        )

    def _floor_min_z(self, q_by_motor: dict[int, float]) -> float:
        if self.floor_model is None or self.floor_data is None or self.floor_mujoco is None:
            return float("inf")

        import numpy as np

        model = self.floor_model
        data = self.floor_data
        mujoco = self.floor_mujoco
        for motor_id, qpos_idx in self.floor_qpos_idx_by_motor.items():
            data.qpos[qpos_idx] = float(q_by_motor.get(motor_id, 0.0))
        mujoco.mj_forward(model, data)

        min_z = float("inf")
        for geom_id in self.floor_checked_geom_ids:
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            vert_adr = int(model.mesh_vertadr[mesh_id])
            vert_num = int(model.mesh_vertnum[mesh_id])
            if vert_num <= 0:
                continue
            local_vertices = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=float)
            xmat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
            xpos = np.asarray(data.geom_xpos[geom_id], dtype=float)
            world_vertices = local_vertices @ xmat.T + xpos
            geom_min_z = float(np.min(world_vertices[:, 2]))
            min_z = min(min_z, geom_min_z)
        return min_z

    def _validate_floor_clearance_or_raise(self, segments: list[Segment]) -> None:
        if not self.check_floor_clearance:
            return

        sample_count = max(2, int(self.collision_samples_per_segment))
        min_allowed_z = float(self.min_floor_clearance_m)
        for segment in segments:
            for idx in range(sample_count):
                u = idx / max(sample_count - 1, 1)
                alpha, _ = _smoothstep5(u)
                q_sample: dict[int, float] = {}
                for motor_id in self.motor_ids:
                    start_q = float(segment.start.get(motor_id, 0.0))
                    end_q = float(segment.end.get(motor_id, start_q))
                    q_sample[motor_id] = start_q + (end_q - start_q) * alpha
                min_z = self._floor_min_z(q_sample)
                if min_z < min_allowed_z:
                    raise ValueError(
                        f"{segment.label}: floor clearance too low at sampled index "
                        f"{idx}/{sample_count - 1}: min_z={min_z:.4f}m < {min_allowed_z:.4f}m"
                    )

    def _validate_gravity_load_or_raise(self, segments: list[Segment]) -> None:
        if not self.check_gravity_load or not self.gravity_load_limit_by_motor:
            return

        sample_count = max(2, int(self.collision_samples_per_segment))
        for segment in segments:
            for idx in range(sample_count):
                u = idx / max(sample_count - 1, 1)
                alpha, _ = _smoothstep5(u)
                q_sample: dict[int, float] = {}
                for motor_id in self.motor_ids:
                    start_q = float(segment.start.get(motor_id, 0.0))
                    end_q = float(segment.end.get(motor_id, start_q))
                    q_sample[motor_id] = start_q + (end_q - start_q) * alpha
                tau_g = self.gravity.compute_gravity_by_motor(q_sample)
                for motor_id, limit in self.gravity_load_limit_by_motor.items():
                    if motor_id not in tau_g:
                        continue
                    if abs(float(tau_g[motor_id])) > float(limit):
                        raise ValueError(
                            f"{segment.label}: gravity load too high at sampled index "
                            f"{idx}/{sample_count - 1}: motor {motor_id} "
                            f"|tau_g|={abs(float(tau_g[motor_id])):.3f}Nm > {float(limit):.3f}Nm"
                        )

    def _parse_random_ranges(self, random_range_text: str) -> dict[int, tuple[float, float]]:
        try:
            raw = json.loads(random_range_text) if random_range_text else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"random_range_by_motor_json must be valid JSON object: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("random_range_by_motor_json must be a JSON object")

        ranges: dict[int, tuple[float, float]] = {}
        for motor_id in self.random_motor_ids:
            if str(motor_id) in raw:
                value = raw[str(motor_id)]
            elif motor_id in raw:
                value = raw[motor_id]
            else:
                value = self.joint_limits_by_motor.get(
                    motor_id,
                    (-float(self.max_abs_joint_rad), float(self.max_abs_joint_rad)),
                )
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    f"random_range_by_motor_json motor {motor_id} must be [lower, upper]"
                )
            lower = float(value[0])
            upper = float(value[1])
            if lower >= upper:
                raise ValueError(
                    f"random_range_by_motor_json motor {motor_id} lower must be < upper"
                )
            self._validate_target_or_raise(motor_id, lower, f"random_range.motor_{motor_id}.lower")
            self._validate_target_or_raise(motor_id, upper, f"random_range.motor_{motor_id}.upper")
            ranges[motor_id] = (lower, upper)
        return ranges

    def _segment_collision_free(self, start_q: dict[int, float], end_q: dict[int, float]) -> bool:
        sample_count = max(2, int(self.collision_samples_per_segment))
        samples: list[dict[int, float]] = []
        for idx in range(sample_count):
            u = idx / max(sample_count - 1, 1)
            alpha, _ = _smoothstep5(u)
            q_sample: dict[int, float] = {}
            for motor_id in self.motor_ids:
                start = float(start_q.get(motor_id, 0.0))
                end = float(end_q.get(motor_id, start))
                q_sample[motor_id] = start + (end - start) * alpha
            samples.append(q_sample)

        if self.collision_checker is not None:
            any_collision, _first_idx = self.collision_checker.check_trajectory(samples)
            if any_collision:
                return False

        if self.check_floor_clearance:
            min_allowed_z = float(self.min_floor_clearance_m)
            for q_sample in samples:
                if self._floor_min_z(q_sample) < min_allowed_z:
                    return False

        if self.check_gravity_load and self.gravity_load_limit_by_motor:
            for q_sample in samples:
                tau_g = self.gravity.compute_gravity_by_motor(q_sample)
                for motor_id, limit in self.gravity_load_limit_by_motor.items():
                    if motor_id in tau_g and abs(float(tau_g[motor_id])) > float(limit):
                        return False

        return True

    def _random_step_norm_ok(self, previous_full_q: dict[int, float], full_q: dict[int, float]) -> bool:
        step_sq = 0.0
        for motor_id in self.random_motor_ids:
            delta = float(full_q.get(motor_id, 0.0)) - float(previous_full_q.get(motor_id, 0.0))
            step_sq += delta * delta
        step_norm = math.sqrt(step_sq)
        return float(self.random_min_step_norm_rad) <= step_norm <= float(self.random_max_step_norm_rad)

    def _generate_random_waypoint_targets(self) -> list[dict[int, float]]:
        rng = random.Random(int(self.random_seed))
        waypoints: list[dict[int, float]] = []
        attempts = 0
        previous_full_q = {motor_id: 0.0 for motor_id in self.motor_ids}
        previous_full_q.update(self.fixed_targets_by_motor)

        while len(waypoints) < int(self.random_waypoint_count) and attempts < int(self.random_max_attempts):
            attempts += 1
            candidate: dict[int, float] = {}
            for motor_id in self.random_motor_ids:
                lower, upper = self.random_ranges_by_motor[motor_id]
                candidate[motor_id] = rng.uniform(lower, upper)

            full_q = dict(previous_full_q)
            full_q.update(self.fixed_targets_by_motor)
            full_q.update(candidate)

            if waypoints and not self._random_step_norm_ok(previous_full_q, full_q):
                continue

            try:
                for motor_id, q_rad in candidate.items():
                    self._validate_target_or_raise(motor_id, q_rad, f"random_waypoint_{len(waypoints) + 1}")
            except ValueError:
                continue

            if not self._segment_collision_free(previous_full_q, full_q):
                continue

            waypoints.append(candidate)
            previous_full_q = full_q

        if len(waypoints) < int(self.random_waypoint_count):
            raise ValueError(
                f"failed to generate {self.random_waypoint_count} collision-free random waypoints "
                f"after {self.random_max_attempts} attempts"
            )

        self.get_logger().info(
            f"generated {len(waypoints)} random waypoints with seed={int(self.random_seed)} "
            f"motors={self.random_motor_ids} ranges={self.random_ranges_by_motor}"
        )
        return waypoints

    def _load_waypoint_targets(self, waypoints_path_text: str) -> list[dict[int, float]]:
        if not waypoints_path_text:
            return []
        path = Path(waypoints_path_text).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        with path.open("r", encoding="utf-8") as waypoint_file:
            raw = json.load(waypoint_file)
        return self._parse_waypoint_targets(raw, f"waypoints_path={path}")

    def _preset_waypoint_targets(self, preset: str) -> list[dict[int, float]]:
        preset_key = preset.strip().lower()
        if not preset_key:
            return []
        if preset_key not in {"j2_fixed_j3_varied", "j3_varied"}:
            raise ValueError("free_motion_preset must be empty, 'j2_fixed_j3_varied', or 'j3_varied'")

        j2 = float(self.free_motion_fixed_j2_rad)
        # Conservative no-contact motion around the range observed to be reachable on hardware.
        return [
            {2: j2, 3: -0.55},
            {2: j2, 3: -1.10},
            {2: j2, 3: -1.45},
            {2: j2, 3: -0.80},
            {2: j2, 3: -1.25},
            {2: j2, 3: -0.60},
        ]

    def _parse_waypoint_targets(self, raw: object, field_name: str) -> list[dict[int, float]]:
        if not isinstance(raw, list):
            raise ValueError(f"{field_name} must contain a JSON list of waypoint objects")

        waypoints: list[dict[int, float]] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"{field_name}[{idx}] must be an object like {{\"2\": -0.825}}")
            waypoint: dict[int, float] = {}
            for key, value in item.items():
                try:
                    motor_id = int(key)
                    q_rad = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{field_name}[{idx}] has invalid target {key}: {value}") from exc
                if motor_id not in self.motor_ids:
                    self.get_logger().warn(f"ignoring waypoint motor not in URDF motor map: {motor_id}")
                    continue
                waypoint[motor_id] = q_rad
            if waypoint:
                waypoints.append(waypoint)
        return waypoints

    def _build_segments(self, initial_q: dict[int, float]) -> list[Segment]:
        if self.waypoint_targets:
            return self._build_waypoint_segments(initial_q)
        if int(self.custom_sweep_motor_id) > 0:
            return self._build_custom_single_joint_segments(initial_q)
        return self._build_default_zero_crossing_segments(initial_q)

    def _build_waypoint_segments(self, initial_q: dict[int, float]) -> list[Segment]:
        segments: list[Segment] = []
        current = dict(initial_q)
        for idx, target in enumerate(self.waypoint_targets):
            end = dict(current)
            end.update(self.fixed_targets_by_motor)
            end.update(target)
            duration_s = self.initial_to_zero_duration_s if idx == 0 else self.segment_duration_s
            if any(abs(current[motor_id] - end[motor_id]) > 1.0e-4 for motor_id in self.required_motor_ids):
                segments.append(
                    Segment(
                        start=dict(current),
                        end=end,
                        duration_s=duration_s,
                        label=f"waypoint_{idx + 1:02d}",
                    )
                )
            current = end
        return segments

    def _build_custom_single_joint_segments(self, initial_q: dict[int, float]) -> list[Segment]:
        sweep_motor_id = int(self.custom_sweep_motor_id)
        start = dict(initial_q)
        high = dict(initial_q)
        high.update(self.fixed_targets_by_motor)
        high[sweep_motor_id] = float(self.custom_sweep_high_rad)
        low = dict(high)
        low[sweep_motor_id] = float(self.custom_sweep_low_rad)

        segments: list[Segment] = []
        if any(abs(start[motor_id] - high[motor_id]) > 1.0e-4 for motor_id in self.required_motor_ids):
            segments.append(
                Segment(
                    start=start,
                    end=high,
                    duration_s=self.initial_to_zero_duration_s,
                    label="current_to_custom_high",
                )
            )
        segments.extend(
            [
                Segment(start=high, end=low, duration_s=self.segment_duration_s, label="custom_high_to_low"),
                Segment(start=low, end=high, duration_s=self.segment_duration_s, label="custom_low_to_high"),
            ]
        )
        return segments

    def _build_default_zero_crossing_segments(self, initial_q: dict[int, float]) -> list[Segment]:
        def waypoint(value: float) -> dict[int, float]:
            q = dict(initial_q)
            q.update(self.fixed_targets_by_motor)
            for motor_id in self.sweep_motor_ids:
                q[motor_id] = float(value)
            return q

        start = dict(initial_q)
        zero = waypoint(self.zero_rad)
        negative = waypoint(self.zero_rad - self.amplitude_rad)
        positive = waypoint(self.zero_rad + self.amplitude_rad)
        segments: list[Segment] = []
        if any(abs(start[motor_id] - zero[motor_id]) > 1.0e-4 for motor_id in self.sweep_motor_ids):
            segments.append(
                Segment(
                    start=start,
                    end=zero,
                    duration_s=self.initial_to_zero_duration_s,
                    label="current_to_zero",
                )
            )
        segments.extend(
            [
                Segment(start=zero, end=negative, duration_s=self.segment_duration_s, label="zero_to_negative"),
                Segment(start=negative, end=zero, duration_s=self.segment_duration_s, label="negative_to_zero"),
                Segment(start=zero, end=positive, duration_s=self.segment_duration_s, label="zero_to_positive"),
                Segment(start=positive, end=zero, duration_s=self.segment_duration_s, label="positive_to_zero"),
            ]
        )
        return segments

    def _validate_sine_motion_or_raise(self) -> None:
        assert self.initial_q_by_motor is not None
        sample_count = max(2, int(self.collision_samples_per_segment))
        duration_s = max(float(self.sine_validation_duration_s), 1.0e-6)
        samples: list[dict[int, float]] = []
        for idx in range(sample_count):
            elapsed_s = duration_s * idx / max(sample_count - 1, 1)
            q_sample, _qd_sample = self._sample_sine_at_elapsed(elapsed_s)
            for motor_id in self.sine_motor_ids:
                self._validate_target_or_raise(
                    motor_id,
                    float(q_sample[motor_id]),
                    f"sine_motion.sample_{idx}.motor_{motor_id}",
                )
            samples.append(q_sample)

        if self.collision_checker is not None:
            any_collision, first_idx = self.collision_checker.check_trajectory(samples)
            if any_collision:
                colliding_pairs = self.collision_checker.colliding_pairs(samples[first_idx])
                raise ValueError(
                    f"sine_motion: self-collision detected at sampled index "
                    f"{first_idx}/{sample_count - 1}; pairs={colliding_pairs[:5]}"
                )

        if self.check_floor_clearance:
            min_allowed_z = float(self.min_floor_clearance_m)
            for idx, q_sample in enumerate(samples):
                min_z = self._floor_min_z(q_sample)
                if min_z < min_allowed_z:
                    raise ValueError(
                        f"sine_motion: floor clearance too low at sampled index "
                        f"{idx}/{sample_count - 1}: min_z={min_z:.4f}m < {min_allowed_z:.4f}m"
                    )

        if self.check_gravity_load and self.gravity_load_limit_by_motor:
            for idx, q_sample in enumerate(samples):
                tau_g = self.gravity.compute_gravity_by_motor(q_sample)
                for motor_id, limit in self.gravity_load_limit_by_motor.items():
                    if motor_id not in tau_g:
                        continue
                    if abs(float(tau_g[motor_id])) > float(limit):
                        raise ValueError(
                            f"sine_motion: gravity load too high at sampled index "
                            f"{idx}/{sample_count - 1}: motor {motor_id} "
                            f"|tau_g|={abs(float(tau_g[motor_id])):.3f}Nm > {float(limit):.3f}Nm"
                        )

    def on_timer(self) -> None:
        now_s = time.monotonic()
        connected_ids = self._connected_motor_ids(now_s)
        connected_tuple = tuple(connected_ids)
        if connected_tuple != self.last_connected_ids:
            self.last_connected_ids = connected_tuple
            self.get_logger().info(f"connected motors updated: {list(connected_tuple)}")

        if not connected_ids:
            if (now_s - self.last_stale_warn_s) >= self.stale_warn_throttle_s:
                self.get_logger().warn(
                    f"no connected motor state within timeout ({self.state_timeout_s:.3f}s): skip publish"
                )
                self.last_stale_warn_s = now_s
            return

        if not self._ensure_sequence_initialized(now_s, connected_ids):
            return

        if self.sine_motion_enabled:
            q_des_by_motor, qd_des_by_motor = self._sample_sine_motion(now_s)
        else:
            q_des_by_motor, qd_des_by_motor = self._sample_sequence(now_s)
        self._publish_command(connected_ids, q_des_by_motor, qd_des_by_motor, now_s)

    def _sine_envelope(self, elapsed_s: float) -> tuple[float, float]:
        if not self.sine_envelope_enabled:
            return 1.0, 0.0
        period_s = max(float(self.sine_envelope_period_s), 1.0e-6)
        theta = 2.0 * math.pi * elapsed_s / period_s
        min_scale = float(self.sine_envelope_min_scale)
        scale = min_scale + (1.0 - min_scale) * 0.5 * (1.0 - math.cos(theta))
        scale_dot = (1.0 - min_scale) * 0.5 * math.sin(theta) * (2.0 * math.pi / period_s)
        return scale, scale_dot

    def _sample_sine_at_elapsed(self, elapsed_s: float) -> tuple[dict[int, float], dict[int, float]]:
        assert self.initial_q_by_motor is not None
        reference_q = self.sine_reference_q_by_motor or self.initial_q_by_motor
        scale, scale_dot = self._sine_envelope(elapsed_s)
        q_des: dict[int, float] = {}
        qd_des: dict[int, float] = {}
        for motor_id in self.motor_ids:
            if motor_id not in self.sine_motor_ids:
                q_des[motor_id] = float(reference_q.get(motor_id, 0.0))
                qd_des[motor_id] = 0.0
                continue
            center = float(self.sine_center_by_motor.get(motor_id, reference_q.get(motor_id, 0.0)))
            amplitude = float(self.sine_amplitude_by_motor.get(motor_id, 0.0))
            frequency_hz = float(self.sine_frequency_by_motor.get(motor_id, 0.0))
            phase = float(self.sine_phase_by_motor.get(motor_id, 0.0))
            omega = 2.0 * math.pi * frequency_hz
            theta = omega * elapsed_s + phase
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            q_des[motor_id] = center + scale * amplitude * sin_theta
            qd_des[motor_id] = amplitude * (scale_dot * sin_theta + scale * omega * cos_theta)
        return q_des, qd_des

    def _sample_sine_motion(self, now_s: float) -> tuple[dict[int, float], dict[int, float]]:
        if not self.sine_warmup_complete:
            q_des, qd_des = self._sample_sequence(now_s)
            if self.sequence_complete:
                self.sine_warmup_complete = True
                self.sine_started_s = now_s
                self.sequence_complete = False
                self.get_logger().info("sine warmup complete; starting continuous sine motion")
                return self._sample_sine_at_elapsed(0.0)
            return q_des, qd_des

        elapsed_s = max(0.0, now_s - self.sine_started_s)
        if self.sine_duration_s > 0.0 and elapsed_s >= self.sine_duration_s:
            if not self.sequence_complete:
                self.get_logger().info("sine motion complete; holding sine centers")
                self.sequence_complete = True
            reference_q = self.sine_reference_q_by_motor or self.initial_q_by_motor
            q_hold: dict[int, float] = {}
            qd_hold: dict[int, float] = {}
            for motor_id in self.motor_ids:
                if motor_id in self.sine_motor_ids:
                    q_hold[motor_id] = float(
                        self.sine_center_by_motor.get(motor_id, reference_q.get(motor_id, 0.0))
                    )
                else:
                    q_hold[motor_id] = float(reference_q.get(motor_id, 0.0))
                qd_hold[motor_id] = 0.0
            return q_hold, qd_hold
        return self._sample_sine_at_elapsed(elapsed_s)

    def _sample_sequence(self, now_s: float) -> tuple[dict[int, float], dict[int, float]]:
        assert self.initial_q_by_motor is not None
        if not self.segments:
            return dict(self.initial_q_by_motor), {motor_id: 0.0 for motor_id in self.motor_ids}

        while self.active_segment_idx < len(self.segments):
            segment = self.segments[self.active_segment_idx]
            elapsed_s = now_s - self.segment_started_s
            if elapsed_s <= segment.duration_s:
                break
            self.active_segment_idx += 1
            self.segment_started_s += segment.duration_s

        if self.active_segment_idx >= len(self.segments):
            if self.repeat:
                final_zero = self.segments[-1].end
                self.segments = self._build_segments(final_zero)
                self.active_segment_idx = 0
                self.segment_started_s = now_s
                self.last_segment_log_idx = -1
            else:
                if not self.sequence_complete:
                    self.get_logger().info("motion sequence complete; holding final waypoint")
                    self.sequence_complete = True
                final_q = self.segments[-1].end
                return dict(final_q), {motor_id: 0.0 for motor_id in self.motor_ids}

        segment = self.segments[self.active_segment_idx]
        if self.active_segment_idx != self.last_segment_log_idx:
            self.last_segment_log_idx = self.active_segment_idx
            self.get_logger().info(
                f"segment {self.active_segment_idx + 1}/{len(self.segments)}: "
                f"{segment.label} duration={segment.duration_s:.2f}s"
            )

        elapsed_s = max(0.0, now_s - self.segment_started_s)
        u = elapsed_s / max(segment.duration_s, 1.0e-6)
        alpha, alpha_du = _smoothstep5(u)

        q_des: dict[int, float] = {}
        qd_des: dict[int, float] = {}
        for motor_id in self.motor_ids:
            start_q = float(segment.start.get(motor_id, 0.0))
            end_q = float(segment.end.get(motor_id, start_q))
            delta_q = end_q - start_q
            q_des[motor_id] = start_q + delta_q * alpha
            qd_des[motor_id] = delta_q * alpha_du / max(segment.duration_s, 1.0e-6)
        return q_des, qd_des

    def _publish_command(
        self,
        connected_ids: list[int],
        q_des_by_motor: dict[int, float],
        qd_des_by_motor: dict[int, float],
        now_s: float,
    ) -> None:
        q_by_motor: dict[int, float] = {}
        for motor_id in self.motor_ids:
            sample = self.state_by_motor[motor_id]
            if math.isfinite(sample.last_seen_s) and (now_s - sample.last_seen_s) <= self.state_timeout_s:
                q_by_motor[motor_id] = sample.q
            else:
                q_by_motor[motor_id] = 0.0
        tau_g_by_motor = self.gravity.compute_gravity_by_motor(q_by_motor)

        stamp = self.get_clock().now().to_msg()
        msg = MotorCMDArray()
        msg.stamp = stamp
        commands: list[MotorCMD] = []
        for motor_id in connected_ids:
            tuning = control_params_for_motor(motor_id)
            gravity_scale = _as_float_or_default(tuning.get("gravity_scale"), 1.0)
            gravity_bias = _as_float_or_default(tuning.get("gravity_bias"), 0.0)
            if self.use_tuning_gains:
                default_kp = self.sweep_kp if motor_id in self.sweep_motor_ids else self.hold_kp
                default_kd = self.sweep_kd if motor_id in self.sweep_motor_ids else self.hold_kd
                kp_value = _as_float_or_default(tuning.get("kp"), default_kp)
                kd_value = _as_float_or_default(tuning.get("kd"), default_kd)
            elif motor_id in self.sweep_motor_ids:
                kp_value = self.sweep_kp
                kd_value = self.sweep_kd
            else:
                kp_value = self.hold_kp
                kd_value = self.hold_kd

            tau_ff = gravity_scale * tau_g_by_motor[motor_id] + gravity_bias
            tau_ff = _clip_symmetric(tau_ff, self.tau_limit_by_motor[motor_id])

            cmd = MotorCMD()
            cmd.stamp = stamp
            cmd.motor_id = int(motor_id)
            cmd.q_des = float(q_des_by_motor[motor_id])
            cmd.qd_des = float(qd_des_by_motor[motor_id])
            cmd.kp = float(kp_value)
            cmd.kd = float(kd_value)
            cmd.tau_ff = float(tau_ff)
            commands.append(cmd)
        msg.commands = commands
        self.cmd_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = JointSweepNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
