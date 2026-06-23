"""Safe random joint motion with gravity-compensated MIT commands.

This node is intentionally joint-space first.  It generates random waypoint
poses inside configurable motor ranges, rejects unsafe segments using
Pinocchio/SRDF self-collision and gravity-load checks, then publishes
``/motor_cmd_array`` with ``q_des``, ``qd_des``, ``kp``, ``kd`` and gravity
feedforward ``tau_ff`` on every tick.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
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
from phy.robot_model import RobotModel
from phy.traj import QuinticPlan, plan_quintic, sample_quintic


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
    label: str


@dataclass
class RuntimeSegment:
    segment: Segment
    plan: QuinticPlan


def _strip(value: object) -> str:
    return str(value).strip()


def _parse_int_list_value(value: object, field_name: str) -> list[int]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be valid JSON list: {exc}") from exc
    else:
        raw = value
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a JSON list or ROS integer array")
    return [int(item) for item in raw]


def _parse_str_list_value(value: object, field_name: str) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be valid JSON list: {exc}") from exc
    else:
        raw = value
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a JSON list or ROS string array")
    return [str(item).strip() for item in raw if str(item).strip()]


def _parse_q_map_json(
    text: str,
    motor_ids: tuple[int, ...],
    default_value: float = 0.0,
) -> dict[int, float]:
    out = {motor_id: float(default_value) for motor_id in motor_ids}
    stripped = str(text).strip()
    if not stripped:
        return out
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"q map must be valid JSON object: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("q map must be an object like {\"2\": -0.7}")
    for key, value in raw.items():
        motor_id = int(key)
        if motor_id in out:
            out[motor_id] = float(value)
    return out


def _parse_range_map_json(
    text: str,
    motor_ids: tuple[int, ...],
    fallback_lower: dict[int, float],
    fallback_upper: dict[int, float],
) -> dict[int, tuple[float, float]]:
    out = {
        motor_id: (float(fallback_lower[motor_id]), float(fallback_upper[motor_id]))
        for motor_id in motor_ids
    }
    stripped = str(text).strip()
    if not stripped:
        return out
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"q range map must be valid JSON object: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("q range map must be an object like {\"2\": [-1.0, 1.57]}")
    for key, value in raw.items():
        motor_id = int(key)
        if motor_id not in out:
            continue
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"q range for motor {motor_id} must be [lower, upper]")
        lo = float(value[0])
        hi = float(value[1])
        if lo > hi:
            lo, hi = hi, lo
        out[motor_id] = (lo, hi)
    return out


def _clip_symmetric(value: float, limit: float) -> float:
    if not math.isfinite(limit) or limit <= 0.0:
        return float(value)
    return float(max(-limit, min(limit, value)))


def _sign_crosses_zero(a: float, b: float, eps: float) -> bool:
    return (
        abs(a) > eps
        and abs(b) > eps
        and ((a > 0.0 and b < 0.0) or (a < 0.0 and b > 0.0))
    )


class RandomMotionGCNode(Node):
    """Generate safe random joint-space motion and publish gravity-comp commands."""

    def __init__(self) -> None:
        super().__init__("random_motion_gc_node")

        self.control_hz = max(1.0, declare_typed(self, "control_hz", 100.0))
        self.state_timeout_s = declare_typed(self, "state_timeout_s", 0.2)
        self.stale_warn_throttle_s = declare_typed(self, "stale_warn_throttle_s", 2.0)
        self.open_loop_preview = declare_typed(self, "open_loop_preview", False)

        self.random_seed = declare_typed(self, "random_seed", 7)
        self.random_waypoint_count = max(1, declare_typed(self, "random_waypoint_count", 12))
        self.max_generation_attempts = max(1, declare_typed(self, "max_generation_attempts", 5000))
        self.start_from_current_state = declare_typed(self, "start_from_current_state", True)
        self.move_to_home_first = declare_typed(self, "move_to_home_first", True)
        self.repeat = declare_typed(self, "repeat", True)

        self.initial_to_home_duration_s = max(
            0.5, declare_typed(self, "initial_to_home_duration_s", 8.0)
        )
        self.segment_min_duration_s = max(0.5, declare_typed(self, "segment_min_duration_s", 8.0))
        self.v_max = max(1.0e-3, declare_typed(self, "v_max", 0.35))
        self.a_max = max(1.0e-3, declare_typed(self, "a_max", 0.70))

        self.kp = declare_typed(self, "kp", 4.0)
        self.kd = declare_typed(self, "kd", 0.8)
        self.hold_kp = declare_typed(self, "hold_kp", 3.0)
        self.hold_kd = declare_typed(self, "hold_kd", 0.5)
        self.use_tuning_gains = declare_typed(self, "use_tuning_gains", False)
        self.unlimited_tau = declare_typed(self, "unlimited_tau", False)

        self.max_abs_joint_rad = declare_typed(self, "max_abs_joint_rad", math.pi)
        self.max_joint_step_rad = declare_typed(self, "max_joint_step_rad", 0.75)
        self.min_joint_step_norm_rad = declare_typed(self, "min_joint_step_norm_rad", 0.08)
        self.enforce_zero_crossing = declare_typed(self, "enforce_zero_crossing", True)
        self.zero_crossing_epsilon_rad = declare_typed(
            self, "zero_crossing_epsilon_rad", 1.0e-4
        )
        self.check_self_collision = declare_typed(self, "check_self_collision", True)
        self.check_frame_z = declare_typed(self, "check_frame_z", True)
        self.min_frame_z_m = declare_typed(self, "min_frame_z_m", 0.02)
        self.check_gravity_load = declare_typed(self, "check_gravity_load", True)
        self.gravity_load_ratio_limit = declare_typed(self, "gravity_load_ratio_limit", 0.95)
        self.collision_samples_per_segment = max(
            2, declare_typed(self, "collision_samples_per_segment", 24)
        )

        default_map_json = json.dumps(DEFAULT_MOTOR_JOINT_MAP)
        default_tau_json = json.dumps(DEFAULT_TAU_LIMIT_BY_MOTOR)
        motor_joint_map_text = declare_typed(
            self, "motor_joint_map_json", default_map_json, cast=_strip
        )
        tau_limit_text = declare_typed(
            self, "tau_limit_by_motor_json", default_tau_json, cast=_strip
        )
        command_ids_value = declare_typed(
            self, "command_motor_ids_json", [1, 2, 3], cast=lambda value: value
        )
        random_ids_value = declare_typed(
            self, "random_motor_ids_json", [1, 2, 3], cast=lambda value: value
        )
        home_q_text = declare_typed(
            self,
            "home_q_by_motor_json",
            '{"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0}',
            cast=_strip,
        )
        q_range_text = declare_typed(
            self,
            "q_range_by_motor_json",
            (
                '{"1": [-0.70, 0.70], "2": [-1.20, 1.57], '
                '"3": [-1.50, 0.05], "4": [-0.05, 0.05], '
                '"5": [-0.05, 0.05], "6": [-0.05, 0.05]}'
            ),
            cast=_strip,
        )
        sample_range_text = declare_typed(
            self,
            "sample_q_range_by_motor_json",
            (
                '{"1": [-0.70, 0.70], "2": [-1.20, 1.57], '
                '"3": [-1.50, -0.40]}'
            ),
            cast=_strip,
        )
        gravity_scale_text = declare_typed(
            self, "gravity_scale_by_motor_json", "{}", cast=_strip
        )
        gravity_bias_text = declare_typed(self, "gravity_bias_by_motor_json", "{}", cast=_strip)
        urdf_path_text = declare_typed(self, "urdf_path", "", cast=_strip)
        srdf_path_text = declare_typed(self, "srdf_path", "", cast=_strip)
        frame_z_names_value = declare_typed(
            self,
            "frame_z_names_json",
            ["link1_1", "link2_1", "link3_1", "link3_2", "link3_3", "gripper"],
            cast=lambda value: value,
        )

        motor_joint_map = parse_motor_joint_map_json(motor_joint_map_text)
        if not motor_joint_map:
            motor_joint_map = dict(DEFAULT_MOTOR_JOINT_MAP)
        self.tau_limit_by_motor = {
            **DEFAULT_TAU_LIMIT_BY_MOTOR,
            **parse_float_map_json(tau_limit_text, "tau_limit_by_motor_json"),
        }
        self.gravity_scale_by_motor = parse_float_map_json(
            gravity_scale_text, "gravity_scale_by_motor_json"
        )
        self.gravity_bias_by_motor = parse_float_map_json(
            gravity_bias_text, "gravity_bias_by_motor_json"
        )

        urdf_path = resolve_share_file("sim", "urdf/robot.urdf", urdf_path_text)
        srdf_path = resolve_share_file("sim", "srdf/robot.srdf", srdf_path_text)
        sim_share_parent = str(Path(get_package_share_directory("sim")).parent)

        self.robot = RobotModel(urdf_path, motor_joint_map)
        self.all_motor_ids = self.robot.ordered_motor_ids
        self.command_motor_ids = tuple(
            mid for mid in _parse_int_list_value(command_ids_value, "command_motor_ids_json")
            if mid in self.robot.bindings
        )
        if not self.command_motor_ids:
            raise ValueError("command_motor_ids_json resolved to no known motors")
        self.random_motor_ids = tuple(
            mid for mid in _parse_int_list_value(random_ids_value, "random_motor_ids_json")
            if mid in self.robot.bindings
        )
        if not self.random_motor_ids:
            raise ValueError("random_motor_ids_json resolved to no known motors")

        self.home_q = _parse_q_map_json(home_q_text, self.all_motor_ids, default_value=0.0)
        lower, upper = self.robot.joint_limits()
        fallback_lower = {
            motor_id: max(float(lower[idx]), -abs(float(self.max_abs_joint_rad)))
            for idx, motor_id in enumerate(self.all_motor_ids)
        }
        fallback_upper = {
            motor_id: min(float(upper[idx]), abs(float(self.max_abs_joint_rad)))
            for idx, motor_id in enumerate(self.all_motor_ids)
        }
        self.q_range_by_motor = _parse_range_map_json(
            q_range_text, self.all_motor_ids, fallback_lower, fallback_upper
        )
        self._clamp_ranges_to_limits(fallback_lower, fallback_upper)
        self.sample_q_range_by_motor = _parse_range_map_json(
            sample_range_text,
            self.all_motor_ids,
            {mid: self.q_range_by_motor[mid][0] for mid in self.all_motor_ids},
            {mid: self.q_range_by_motor[mid][1] for mid in self.all_motor_ids},
        )
        self._clamp_sample_ranges_to_safety_ranges()
        requested_frame_z_names = _parse_str_list_value(
            frame_z_names_value, "frame_z_names_json"
        )
        available_frame_names = {frame.name for frame in self.robot.model.frames}
        self.frame_z_names = tuple(
            name for name in requested_frame_z_names if name in available_frame_names
        )
        missing_frame_z_names = [
            name for name in requested_frame_z_names if name not in available_frame_names
        ]
        if missing_frame_z_names:
            self.get_logger().warn(
                f"frame_z_names_json contains missing URDF frames; ignoring {missing_frame_z_names}"
            )
        if self.check_frame_z and not self.frame_z_names:
            self.get_logger().warn("check_frame_z enabled but no valid frames remain; disabling")
            self.check_frame_z = False

        self.collision_checker: Optional[CollisionChecker] = None
        if self.check_self_collision:
            self.collision_checker = CollisionChecker(
                self.robot,
                srdf_path=srdf_path,
                package_dirs=[sim_share_parent],
            )

        self.rng = np.random.default_rng(int(self.random_seed))
        self.state_by_motor = {motor_id: MotorSample() for motor_id in self.all_motor_ids}
        self.runtime_segments: list[RuntimeSegment] = []
        self.active_segment_idx = 0
        self.segment_start_s: Optional[float] = None
        self._hold_q: Optional[dict[int, float]] = None
        self._sequence_initialized = False
        self._sequence_complete = False
        self._last_stale_warn_s = float("-inf")

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
        self.cmd_pub = self.create_publisher(MotorCMDArray, "/motor_cmd_array", qos_cmd)
        self.state_sub = self.create_subscription(
            MotorStateArray, "/motor_state_array", self.on_state_array, qos_state
        )
        self.timer = self.create_timer(max(1.0 / self.control_hz, 1.0e-4), self.on_timer)

        self.get_logger().info(
            "random_motion_gc_node initialized: "
            f"hz={self.control_hz:.1f} command_motors={list(self.command_motor_ids)} "
            f"random_motors={list(self.random_motor_ids)} waypoints={self.random_waypoint_count} "
            f"home={self.home_q} safety_ranges={self.q_range_by_motor} "
            f"sample_ranges={self.sample_q_range_by_motor} "
            f"self_collision={self.check_self_collision} gravity_load={self.check_gravity_load} "
            f"frame_z={self.check_frame_z} min_frame_z={self.min_frame_z_m:.3f} "
            f"frame_z_names={list(self.frame_z_names)} "
            f"urdf={urdf_path}"
        )

    def on_state_array(self, msg: MotorStateArray) -> None:
        stamp_s = float(msg.stamp.sec) + float(msg.stamp.nanosec) * 1.0e-9
        for state in msg.states:
            motor_id = int(state.motor_id)
            if motor_id not in self.state_by_motor:
                continue
            self.state_by_motor[motor_id] = MotorSample(
                q=float(state.q),
                qd=float(state.qd),
                tau_measured=float(state.tau),
                last_seen_s=stamp_s,
            )

    def on_timer(self) -> None:
        now_s = self._now_s()
        if not self._sequence_initialized:
            start_q = self._initial_q(now_s)
            if start_q is None:
                return
            self.runtime_segments = self._build_runtime_sequence(start_q)
            self.active_segment_idx = 0
            self.segment_start_s = now_s
            self._sequence_initialized = True
            self.get_logger().info(
                f"random motion sequence ready: segments={len(self.runtime_segments)}"
            )

        if not self.runtime_segments:
            return
        if self._sequence_complete:
            assert self._hold_q is not None
            zero_qd = {motor_id: 0.0 for motor_id in self.all_motor_ids}
            self._publish(self._hold_q, zero_qd, hold=True)
            return

        runtime = self.runtime_segments[self.active_segment_idx]
        assert self.segment_start_s is not None
        elapsed_s = now_s - self.segment_start_s
        q_vec, qd_vec, done = sample_quintic(runtime.plan, elapsed_s)
        q_by_motor = self._vec_to_q_dict(q_vec)
        qd_by_motor = self._vec_to_q_dict(qd_vec)
        self._publish(q_by_motor, qd_by_motor, hold=False)

        if done:
            self.get_logger().info(f"segment done: {runtime.segment.label}")
            self.active_segment_idx += 1
            if self.active_segment_idx >= len(self.runtime_segments):
                if self.repeat:
                    self.active_segment_idx = 0
                    self.get_logger().info("random motion sequence repeat")
                else:
                    self.active_segment_idx = len(self.runtime_segments) - 1
                    self._hold_q = dict(runtime.segment.end)
                    self._sequence_complete = True
                    self.get_logger().info("random motion sequence complete — holding final pose")
            self.segment_start_s = now_s

    def _initial_q(self, now_s: float) -> Optional[dict[int, float]]:
        if self.open_loop_preview or not self.start_from_current_state:
            return dict(self.home_q)
        if not self._state_fresh(now_s):
            if now_s - self._last_stale_warn_s >= self.stale_warn_throttle_s:
                missing = [
                    mid
                    for mid in self.command_motor_ids
                    if not math.isfinite(self.state_by_motor[mid].last_seen_s)
                    or (now_s - self.state_by_motor[mid].last_seen_s) > self.state_timeout_s
                ]
                self.get_logger().warn(
                    f"waiting for fresh /motor_state_array: missing/stale motors={missing}"
                )
                self._last_stale_warn_s = now_s
            return None
        q = dict(self.home_q)
        for motor_id in self.command_motor_ids:
            q[motor_id] = float(self.state_by_motor[motor_id].q)
        return q

    def _build_runtime_sequence(self, start_q: dict[int, float]) -> list[RuntimeSegment]:
        segments = self._build_joint_segments(start_q)
        runtime: list[RuntimeSegment] = []
        for segment in segments:
            q0 = self._q_dict_to_vec(segment.start)
            q1 = self._q_dict_to_vec(segment.end)
            min_duration = (
                self.initial_to_home_duration_s
                if segment.label == "current_to_home"
                else self.segment_min_duration_s
            )
            plan = plan_quintic(
                q_start=q0,
                q_goal=q1,
                v_start=np.zeros_like(q0),
                v_goal=np.zeros_like(q1),
                v_max=np.full_like(q0, self.v_max),
                a_max=np.full_like(q0, self.a_max),
                min_duration=min_duration,
            )
            runtime.append(RuntimeSegment(segment=segment, plan=plan))
        return runtime

    def _build_joint_segments(self, start_q: dict[int, float]) -> list[Segment]:
        self._assert_q_safe(start_q, "start_q")
        segments: list[Segment] = []
        current = dict(start_q)

        if self.move_to_home_first and self._q_distance(current, self.home_q) > 1.0e-4:
            for end in self._expand_and_split_path(current, self.home_q):
                label = "current_to_home" if not segments else "current_to_home_zero"
                self._assert_segment_safe(current, end, label)
                segments.append(Segment(start=dict(current), end=dict(end), label=label))
                current = dict(end)

        accepted = 0
        attempts = 0
        while accepted < self.random_waypoint_count and attempts < self.max_generation_attempts:
            attempts += 1
            candidate = self._sample_random_q()
            if self._q_distance(current, candidate) < self.min_joint_step_norm_rad:
                continue

            expanded = self._expand_and_split_path(current, candidate)
            if not self._path_safe(current, expanded):
                continue

            for end in expanded:
                label = f"waypoint_{accepted + 1:03d}"
                if end is not expanded[-1]:
                    label += "_zero_crossing"
                segments.append(Segment(start=dict(current), end=dict(end), label=label))
                current = dict(end)
            accepted += 1

        if accepted < self.random_waypoint_count:
            raise RuntimeError(
                f"generated only {accepted}/{self.random_waypoint_count} safe waypoints "
                f"after {attempts} attempts; relax ranges or constraints"
            )
        self.get_logger().info(
            f"generated {accepted} safe random waypoints after {attempts} attempts"
        )
        return segments

    def _sample_random_q(self) -> dict[int, float]:
        q = dict(self.home_q)
        for motor_id in self.random_motor_ids:
            lo, hi = self.sample_q_range_by_motor[motor_id]
            q[motor_id] = float(self.rng.uniform(lo, hi))
        return q

    def _expand_zero_crossing(
        self, start: dict[int, float], goal: dict[int, float]
    ) -> list[dict[int, float]]:
        if not self.enforce_zero_crossing:
            return [dict(goal)]
        crossing = [
            mid
            for mid in self.random_motor_ids
            if _sign_crosses_zero(
                start[mid], goal[mid], float(self.zero_crossing_epsilon_rad)
            )
        ]
        if not crossing:
            return [dict(goal)]
        mid_q = dict(goal)
        for motor_id in crossing:
            mid_q[motor_id] = 0.0
        return [mid_q, dict(goal)]

    def _expand_and_split_path(
        self, start: dict[int, float], goal: dict[int, float]
    ) -> list[dict[int, float]]:
        """Apply zero-crossing waypoints, then split large joint jumps.

        This keeps the "must pass 0 when sign changes" rule while avoiding the
        previous all-or-nothing rejection when the first random target was far
        from home, e.g. j3: 0 -> -0.9 rad.
        """

        out: list[dict[int, float]] = []
        current = dict(start)
        for waypoint in self._expand_zero_crossing(current, goal):
            split_points = self._split_large_step(current, waypoint)
            out.extend(split_points)
            current = dict(split_points[-1])
        return out

    def _split_large_step(
        self, start: dict[int, float], goal: dict[int, float]
    ) -> list[dict[int, float]]:
        max_delta = max(
            abs(float(goal[mid]) - float(start[mid]))
            for mid in self.random_motor_ids
        )
        max_step = max(1.0e-6, float(self.max_joint_step_rad))
        n_steps = max(1, int(math.ceil(max_delta / max_step)))
        points: list[dict[int, float]] = []
        for step_idx in range(1, n_steps + 1):
            alpha = step_idx / n_steps
            points.append(
                {
                    motor_id: (1.0 - alpha) * float(start[motor_id])
                    + alpha * float(goal[motor_id])
                    for motor_id in self.all_motor_ids
                }
            )
        return points

    def _path_safe(self, start: dict[int, float], waypoints: list[dict[int, float]]) -> bool:
        current = dict(start)
        for idx, end in enumerate(waypoints):
            label = f"path_candidate_{idx}"
            try:
                self._assert_segment_safe(current, end, label)
            except ValueError:
                return False
            current = dict(end)
        return True

    def _assert_q_safe(self, q: dict[int, float], label: str) -> None:
        for motor_id in self.all_motor_ids:
            value = float(q[motor_id])
            lo, hi = self.q_range_by_motor[motor_id]
            if value < lo - 1.0e-6 or value > hi + 1.0e-6:
                raise ValueError(
                    f"{label}: motor {motor_id} q={value:.4f} outside range [{lo:.4f}, {hi:.4f}]"
                )
            if abs(value) > abs(float(self.max_abs_joint_rad)) + 1.0e-6:
                raise ValueError(f"{label}: motor {motor_id} q={value:.4f} exceeds max_abs_joint_rad")

        if self.collision_checker is not None and self.collision_checker.check(q):
            pairs = self.collision_checker.colliding_pairs(q)
            raise ValueError(f"{label}: self-collision detected pairs={pairs[:5]}")
        if self.check_frame_z:
            self._assert_frame_z_safe(q, label)
        if self.check_gravity_load:
            self._assert_gravity_load_safe(q, label)

    def _assert_segment_safe(
        self, start: dict[int, float], end: dict[int, float], label: str
    ) -> None:
        samples = self._sample_segment_qs(start, end, self.collision_samples_per_segment)
        for idx, q in enumerate(samples):
            self._assert_joint_range_only(q, f"{label}[{idx}]")

        if self.collision_checker is not None:
            any_collision, first_idx = self.collision_checker.check_trajectory(samples)
            if any_collision:
                pairs = self.collision_checker.colliding_pairs(samples[first_idx])
                raise ValueError(
                    f"{label}: self-collision at sample {first_idx}/{len(samples) - 1} "
                    f"pairs={pairs[:5]}"
                )
        if self.check_gravity_load:
            for idx, q in enumerate(samples):
                self._assert_gravity_load_safe(q, f"{label}[{idx}]")
        if self.check_frame_z:
            for idx, q in enumerate(samples):
                self._assert_frame_z_safe(q, f"{label}[{idx}]")

    def _assert_joint_range_only(self, q: dict[int, float], label: str) -> None:
        for motor_id, value in q.items():
            lo, hi = self.q_range_by_motor[motor_id]
            if value < lo - 1.0e-6 or value > hi + 1.0e-6:
                raise ValueError(
                    f"{label}: motor {motor_id} q={value:.4f} outside range [{lo:.4f}, {hi:.4f}]"
                )
            if abs(value) > abs(float(self.max_abs_joint_rad)) + 1.0e-6:
                raise ValueError(f"{label}: motor {motor_id} q={value:.4f} exceeds max_abs_joint_rad")

    def _assert_gravity_load_safe(self, q: dict[int, float], label: str) -> None:
        tau_ff = self._tau_ff_by_motor(q)
        for motor_id in self.command_motor_ids:
            limit = float(self.tau_limit_by_motor.get(motor_id, float("inf")))
            if not math.isfinite(limit) or limit <= 0.0:
                continue
            allowed = limit * float(self.gravity_load_ratio_limit)
            if abs(tau_ff[motor_id]) > allowed:
                raise ValueError(
                    f"{label}: gravity tau motor {motor_id} {tau_ff[motor_id]:+.3f}Nm "
                    f"> {allowed:.3f}Nm"
                )

    def _assert_frame_z_safe(self, q: dict[int, float], label: str) -> None:
        min_z = float("inf")
        min_frame = ""
        for frame_name in self.frame_z_names:
            z = float(self.robot.forward_kinematics(q, frame_name).translation[2])
            if z < min_z:
                min_z = z
                min_frame = frame_name
        if min_z < float(self.min_frame_z_m):
            raise ValueError(
                f"{label}: frame '{min_frame}' z={min_z:.4f}m "
                f"< min_frame_z_m={float(self.min_frame_z_m):.4f}m"
            )

    def _sample_segment_qs(
        self, start: dict[int, float], end: dict[int, float], n: int
    ) -> list[dict[int, float]]:
        out: list[dict[int, float]] = []
        denom = max(n - 1, 1)
        for idx in range(n):
            alpha = idx / denom
            out.append(
                {
                    motor_id: (1.0 - alpha) * float(start[motor_id]) + alpha * float(end[motor_id])
                    for motor_id in self.all_motor_ids
                }
            )
        return out

    def _publish(
        self,
        q_by_motor: dict[int, float],
        qd_by_motor: dict[int, float],
        hold: bool,
    ) -> None:
        tau_ff_by_motor = self._tau_ff_by_motor(q_by_motor)
        stamp = self.get_clock().now().to_msg()
        msg = MotorCMDArray()
        msg.stamp = stamp
        commands: list[MotorCMD] = []
        for motor_id in self.command_motor_ids:
            kp, kd = self._gains_for_motor(motor_id, hold=hold)
            tau_limit = float(self.tau_limit_by_motor.get(motor_id, float("inf")))
            tau_ff = tau_ff_by_motor[motor_id]
            if not self.unlimited_tau:
                tau_ff = _clip_symmetric(tau_ff, tau_limit)
            if not all(
                math.isfinite(v)
                for v in (q_by_motor[motor_id], qd_by_motor[motor_id], kp, kd, tau_ff)
            ):
                self.get_logger().warn(f"non-finite command for motor {motor_id}; dropping tick")
                return
            cmd = MotorCMD()
            cmd.stamp = stamp
            cmd.motor_id = int(motor_id)
            cmd.q_des = float(q_by_motor[motor_id])
            cmd.qd_des = float(qd_by_motor[motor_id])
            cmd.kp = float(kp)
            cmd.kd = float(kd)
            cmd.tau_ff = float(tau_ff)
            commands.append(cmd)
        msg.commands = commands
        self.cmd_pub.publish(msg)

    def _tau_ff_by_motor(self, q_by_motor: dict[int, float]) -> dict[int, float]:
        tau_g = self.robot.gravity_torque(q_by_motor)
        out: dict[int, float] = {}
        for motor_id in self.all_motor_ids:
            if self.use_tuning_gains:
                tuning = control_params_for_motor(motor_id)
                scale = float(tuning.get("gravity_scale", 1.0))
                bias = float(tuning.get("gravity_bias", 0.0))
            else:
                scale = float(self.gravity_scale_by_motor.get(motor_id, 1.0))
                bias = float(self.gravity_bias_by_motor.get(motor_id, 0.0))
            out[motor_id] = scale * float(tau_g[motor_id]) + bias
        return out

    def _gains_for_motor(self, motor_id: int, hold: bool) -> tuple[float, float]:
        if self.use_tuning_gains:
            tuning = control_params_for_motor(motor_id)
            return float(tuning.get("kp", 0.0)), float(tuning.get("kd", 0.0))
        if hold:
            return float(self.hold_kp), float(self.hold_kd)
        return float(self.kp), float(self.kd)

    def _state_fresh(self, now_s: float) -> bool:
        for motor_id in self.command_motor_ids:
            sample = self.state_by_motor[motor_id]
            if (
                not math.isfinite(sample.last_seen_s)
                or (now_s - sample.last_seen_s) > self.state_timeout_s
            ):
                return False
        return True

    def _clamp_ranges_to_limits(
        self, fallback_lower: dict[int, float], fallback_upper: dict[int, float]
    ) -> None:
        for motor_id in self.all_motor_ids:
            lo, hi = self.q_range_by_motor[motor_id]
            lo = max(lo, fallback_lower[motor_id], -abs(float(self.max_abs_joint_rad)))
            hi = min(hi, fallback_upper[motor_id], abs(float(self.max_abs_joint_rad)))
            if lo > hi:
                raise ValueError(
                    f"q range for motor {motor_id} becomes empty after joint-limit clamp: "
                    f"[{lo:.4f}, {hi:.4f}]"
                )
            self.q_range_by_motor[motor_id] = (lo, hi)
            if not (lo - 1.0e-6 <= self.home_q[motor_id] <= hi + 1.0e-6):
                raise ValueError(
                    f"home_q motor {motor_id}={self.home_q[motor_id]:.4f} outside "
                    f"range [{lo:.4f}, {hi:.4f}]"
                )

    def _clamp_sample_ranges_to_safety_ranges(self) -> None:
        for motor_id in self.all_motor_ids:
            sample_lo, sample_hi = self.sample_q_range_by_motor[motor_id]
            safe_lo, safe_hi = self.q_range_by_motor[motor_id]
            lo = max(sample_lo, safe_lo)
            hi = min(sample_hi, safe_hi)
            if lo > hi:
                raise ValueError(
                    f"sample range for motor {motor_id} becomes empty after safety clamp: "
                    f"[{lo:.4f}, {hi:.4f}]"
                )
            self.sample_q_range_by_motor[motor_id] = (lo, hi)

    def _q_dict_to_vec(self, q: dict[int, float]) -> np.ndarray:
        return np.asarray([q[motor_id] for motor_id in self.all_motor_ids], dtype=float)

    def _vec_to_q_dict(self, q_vec: np.ndarray) -> dict[int, float]:
        return {
            motor_id: float(q_vec[idx])
            for idx, motor_id in enumerate(self.all_motor_ids)
        }

    def _q_distance(self, a: dict[int, float], b: dict[int, float]) -> float:
        return float(
            np.linalg.norm(
                np.asarray(
                    [float(a[mid]) - float(b[mid]) for mid in self.random_motor_ids],
                    dtype=float,
                )
            )
        )

    def _max_random_motor_step(self, a: dict[int, float], b: dict[int, float]) -> float:
        return max(abs(float(a[mid]) - float(b[mid])) for mid in self.random_motor_ids)

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = RandomMotionGCNode()
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
