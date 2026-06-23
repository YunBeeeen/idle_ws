"""Dataset free-motion command node using IK-generated safe waypoints.

This node is intentionally independent from MuJoCo launch files.  It only
subscribes to ``/motor_state_array`` and publishes ``/motor_cmd_array`` so the
same command generator can be used with either the real CAN bridge or a
separately launched MuJoCo bridge/viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

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
from phy.gravity import GravityCompensator
from phy.ik import IKConfig, IKPolicyConfig, IKSolver
from phy.robot_model import RobotModel
from phy.traj import QuinticPlan, plan_quintic, sample_quintic


@dataclass
class MotorSample:
    q: float = 0.0
    qd: float = 0.0
    tau_measured: float = 0.0
    last_seen_s: float = float("-inf")


@dataclass(frozen=True)
class JointSegment:
    start: dict[int, float]
    end: dict[int, float]
    label: str


@dataclass
class RuntimeSegment:
    segment: JointSegment
    plan: QuinticPlan


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


def _parse_float_triplet_json(text: str, field_name: str) -> tuple[float, float, float]:
    try:
        raw = json.loads(str(text).strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON list: {exc}") from exc
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{field_name} must be [x, y, z]")
    return float(raw[0]), float(raw[1]), float(raw[2])


def _parse_full_q_json(
    text: str,
    motor_ids: tuple[int, ...],
    default: dict[int, float] | None = None,
) -> dict[int, float]:
    out = {motor_id: 0.0 for motor_id in motor_ids}
    if default:
        out.update({int(k): float(v) for k, v in default.items() if int(k) in out})
    stripped = str(text).strip()
    if not stripped:
        return out
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"q JSON must be valid object: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("q JSON must be an object like {\"2\": -0.7}")
    for key, value in raw.items():
        motor_id = int(key)
        if motor_id in out:
            out[motor_id] = float(value)
    return out


def _sign_crosses_zero(a: float, b: float, eps: float) -> bool:
    return abs(a) > eps and abs(b) > eps and ((a > 0.0 and b < 0.0) or (a < 0.0 and b > 0.0))


class DatasetIkMotionNode(Node):
    """Generate safe IK free-motion waypoints and publish MIT commands."""

    def __init__(self) -> None:
        super().__init__("dataset_ik_motion_node")

        strip_str = lambda value: str(value).strip()

        self.control_hz = declare_typed(self, "control_hz", 100.0)
        self.state_timeout_s = declare_typed(self, "state_timeout_s", 0.2)
        self.stale_warn_throttle_s = declare_typed(self, "stale_warn_throttle_s", 2.0)
        self.open_loop_preview = declare_typed(self, "open_loop_preview", False)
        self.kp = declare_typed(self, "kp", 4.0)
        self.kd = declare_typed(self, "kd", 0.8)
        self.hold_kp = declare_typed(self, "hold_kp", 3.0)
        self.hold_kd = declare_typed(self, "hold_kd", 0.5)
        self.use_tuning_gains = declare_typed(self, "use_tuning_gains", False)
        self.start_all_motors_at_zero = declare_typed(self, "start_all_motors_at_zero", True)
        self.move_to_workspace_center_first = declare_typed(self, "move_to_workspace_center_first", True)
        self.initial_to_zero_duration_s = max(0.5, declare_typed(self, "initial_to_zero_duration_s", 10.0))
        self.segment_duration_s = max(0.5, declare_typed(self, "segment_duration_s", 8.0))
        self.repeat = declare_typed(self, "repeat", False)

        self.random_seed = declare_typed(self, "random_seed", 11)
        self.random_waypoint_count = max(1, declare_typed(self, "random_waypoint_count", 12))
        self.max_generation_attempts = max(1, declare_typed(self, "max_generation_attempts", 3000))
        self.workspace_delta_xyz = np.asarray(
            _parse_float_triplet_json(
                declare_typed(self, "workspace_delta_xyz_json", "[0.06, 0.08, 0.05]", cast=strip_str),
                "workspace_delta_xyz_json",
            ),
            dtype=float,
        )
        self.workspace_bias_xyz = np.asarray(
            _parse_float_triplet_json(
                declare_typed(self, "workspace_bias_xyz_json", "[0.0, 0.0, 0.0]", cast=strip_str),
                "workspace_bias_xyz_json",
            ),
            dtype=float,
        )

        self.v_max = max(1.0e-3, declare_typed(self, "v_max", 0.35))
        self.a_max = max(1.0e-3, declare_typed(self, "a_max", 0.70))
        self.max_abs_joint_rad = declare_typed(self, "max_abs_joint_rad", math.pi)
        self.enforce_zero_crossing = declare_typed(self, "enforce_zero_crossing", True)
        self.zero_crossing_epsilon_rad = declare_typed(self, "zero_crossing_epsilon_rad", 1.0e-4)
        self.check_self_collision = declare_typed(self, "check_self_collision", True)
        self.check_gravity_load = declare_typed(self, "check_gravity_load", True)
        self.collision_samples_per_segment = max(2, declare_typed(self, "collision_samples_per_segment", 16))
        self.max_ik_residual_accept_m = declare_typed(self, "max_ik_residual_accept_m", 0.010)
        self.ik_random_restarts = declare_typed(self, "ik_random_restarts", 32)
        self.ik_seed_default_span = declare_typed(self, "ik_seed_default_span", math.pi)
        self.max_joint_jump_rad = declare_typed(self, "max_joint_jump_rad", 0.75)
        self.min_joint_step_norm_rad = declare_typed(self, "min_joint_step_norm_rad", 0.08)
        self.min_per_motion_joint_step_rad = declare_typed(self, "min_per_motion_joint_step_rad", 0.04)
        self.goal_sampling_mode = declare_typed(self, "goal_sampling_mode", "joint_fk", cast=strip_str)

        default_map_json = json.dumps(DEFAULT_MOTOR_JOINT_MAP)
        default_limit_json = json.dumps(DEFAULT_TAU_LIMIT_BY_MOTOR)
        map_json_text = declare_typed(self, "motor_joint_map_json", default_map_json, cast=strip_str)
        limit_json_text = declare_typed(self, "tau_limit_by_motor_json", default_limit_json, cast=strip_str)
        controlled_ids_value = declare_typed(
            self, "controlled_motor_ids_json", [1, 2, 3, 4, 5, 6], cast=lambda value: value
        )
        require_motion_ids_value = declare_typed(
            self, "require_motion_motor_ids_json", [1, 2, 3], cast=lambda value: value
        )
        joint_delta_text = declare_typed(
            self,
            "joint_sample_delta_by_motor_json",
            '{"1": 0.22, "2": 0.18, "3": 0.22}',
            cast=strip_str,
        )
        urdf_path_text = declare_typed(self, "urdf_path", "", cast=strip_str)
        srdf_path_text = declare_typed(self, "srdf_path", "", cast=strip_str)
        target_frame = declare_typed(self, "target_frame", "gripper", cast=strip_str)
        target_offset_json = declare_typed(self, "target_offset_xyz_json", "[0.0, 0.0, 0.0]", cast=strip_str)
        workspace_center_q_json = declare_typed(
            self,
            "workspace_center_q_by_motor_json",
            '{"1": 0.0, "2": -0.70, "3": -1.20, "4": 0.0, "5": 0.0, "6": 0.0}',
            cast=strip_str,
        )

        ik_max_iterations = declare_typed(self, "ik_max_iterations", 220)
        ik_tolerance = declare_typed(self, "ik_tolerance", 1.0e-5)
        ik_damping = declare_typed(self, "ik_damping", 1.0e-6)
        ik_step_scale = declare_typed(self, "ik_step_scale", 1.0)

        motor_joint_map = parse_motor_joint_map_json(map_json_text)
        if not motor_joint_map:
            motor_joint_map = dict(DEFAULT_MOTOR_JOINT_MAP)
        tau_limit_map = parse_float_map_json(limit_json_text, "tau_limit_by_motor_json")
        if not tau_limit_map:
            tau_limit_map = dict(DEFAULT_TAU_LIMIT_BY_MOTOR)

        urdf_path = resolve_share_file("sim", "urdf/robot.urdf", urdf_path_text)
        srdf_path = resolve_share_file("sim", "srdf/robot.srdf", srdf_path_text)
        self.robot = RobotModel(urdf_path, motor_joint_map)
        self.gravity = GravityCompensator(urdf_path, motor_joint_map)
        self.motor_joint_map = motor_joint_map
        self.motor_ids = self.robot.ordered_motor_ids

        controlled_ids = [
            motor_id
            for motor_id in _parse_int_list_value(controlled_ids_value, "controlled_motor_ids_json")
            if motor_id in self.motor_ids
        ]
        if not controlled_ids:
            controlled_ids = [motor_id for motor_id in (1, 2, 3) if motor_id in self.motor_ids]
        self.controlled_motor_ids = tuple(sorted(set(controlled_ids)))
        self.controlled_joint_names = tuple(self.motor_joint_map[motor_id] for motor_id in self.controlled_motor_ids)
        self.require_motion_motor_ids = tuple(
            motor_id
            for motor_id in _parse_int_list_value(require_motion_ids_value, "require_motion_motor_ids_json")
            if motor_id in self.controlled_motor_ids
        )
        if not self.require_motion_motor_ids:
            self.require_motion_motor_ids = self.controlled_motor_ids
        self.joint_sample_delta_by_motor = parse_float_map_json(
            joint_delta_text,
            "joint_sample_delta_by_motor_json",
        )

        target_offset = _parse_float_triplet_json(target_offset_json, "target_offset_xyz_json")
        self.ik_solver = IKSolver(
            urdf_path,
            IKConfig(
                target_frame=target_frame,
                target_offset=target_offset,
                controlled_joints=self.controlled_joint_names,
                max_iterations=ik_max_iterations,
                tolerance=ik_tolerance,
                damping=ik_damping,
                step_scale=ik_step_scale,
            ),
        )
        self.ik_policy = IKPolicyConfig(
            max_ik_residual_accept_m=float(self.max_ik_residual_accept_m),
            ik_random_restarts=int(self.ik_random_restarts),
            ik_seed_default_span=float(self.ik_seed_default_span),
            max_joint_jump_rad=float(self.max_joint_jump_rad),
            use_heuristic_seed=True,
        )

        self.collision_checker: CollisionChecker | None = None
        if self.check_self_collision:
            sim_share_parent = str(Path(get_package_share_directory("sim")).parent)
            self.collision_checker = CollisionChecker(
                self.robot,
                srdf_path=srdf_path,
                package_dirs=[sim_share_parent],
            )

        self.workspace_center_q = _parse_full_q_json(workspace_center_q_json, self.motor_ids)
        self.workspace_center_xyz = np.asarray(
            self.robot.forward_kinematics(self.workspace_center_q, target_frame).translation,
            dtype=float,
        ) + self.workspace_bias_xyz

        lower, upper = self.robot.joint_limits()
        self.lower_limit_by_motor = {motor_id: float(lower[idx]) for idx, motor_id in enumerate(self.motor_ids)}
        self.upper_limit_by_motor = {motor_id: float(upper[idx]) for idx, motor_id in enumerate(self.motor_ids)}
        self.tau_limit_by_motor = {
            motor_id: float(tau_limit_map.get(motor_id, float("inf"))) for motor_id in self.motor_ids
        }

        self.state_by_motor = {motor_id: MotorSample() for motor_id in self.motor_ids}
        self.last_connected_ids: tuple[int, ...] = tuple()
        self.last_stale_warn_s = float("-inf")
        self.initialized = False
        self.runtime_segments: list[RuntimeSegment] = []
        self.active_segment_idx = 0
        self.segment_started_s = float("nan")
        self.sequence_complete = False
        self.last_segment_log_idx = -1
        self.current_q_des_by_motor = {motor_id: 0.0 for motor_id in self.motor_ids}
        self.current_qd_des_by_motor = {motor_id: 0.0 for motor_id in self.motor_ids}
        self.rng = np.random.default_rng(int(self.random_seed))

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

        period_s = max(1.0 / float(self.control_hz), 1.0e-4)
        self.timer = self.create_timer(period_s, self.on_timer)

        self.get_logger().info(
            "dataset_ik_motion_node initialized: "
            f"motors={list(self.motor_ids)} controlled={list(self.controlled_motor_ids)} "
            f"require_motion={list(self.require_motion_motor_ids)} "
            f"target_frame={target_frame} center_xyz={self.workspace_center_xyz.tolist()} "
            f"delta_xyz={self.workspace_delta_xyz.tolist()} waypoints={self.random_waypoint_count} "
            f"goal_sampling_mode={self.goal_sampling_mode} open_loop_preview={self.open_loop_preview} "
            f"start_all_zero={self.start_all_motors_at_zero} self_collision={self.check_self_collision} "
            f"gravity_load={self.check_gravity_load} repeat={self.repeat} urdf={urdf_path}"
        )

    def on_state_array(self, msg: MotorStateArray) -> None:
        now_s = time.monotonic()
        for state in msg.states:
            motor_id = int(state.motor_id)
            sample = self.state_by_motor.get(motor_id)
            if sample is None:
                continue
            sample.q = float(state.q)
            sample.qd = float(state.qd)
            sample.tau_measured = float(state.tau)
            sample.last_seen_s = now_s

    def _connected_motor_ids(self, now_s: float) -> list[int]:
        connected: list[int] = []
        for motor_id in self.motor_ids:
            sample = self.state_by_motor[motor_id]
            if math.isfinite(sample.last_seen_s) and (now_s - sample.last_seen_s) <= self.state_timeout_s:
                connected.append(motor_id)
        return connected

    def _full_q_from_state(self) -> dict[int, float]:
        return {motor_id: float(self.state_by_motor[motor_id].q) for motor_id in self.motor_ids}

    def _zero_q(self) -> dict[int, float]:
        return {motor_id: 0.0 for motor_id in self.motor_ids}

    def _ordered_controlled(self, q_by_motor: dict[int, float]) -> np.ndarray:
        return np.asarray([q_by_motor[motor_id] for motor_id in self.controlled_motor_ids], dtype=float)

    def _full_with_controlled(self, base: dict[int, float], q_controlled: np.ndarray) -> dict[int, float]:
        out = dict(base)
        for idx, motor_id in enumerate(self.controlled_motor_ids):
            out[motor_id] = float(q_controlled[idx])
        return out

    def _validate_q_or_raise(self, q_by_motor: dict[int, float], label: str) -> None:
        for motor_id in self.motor_ids:
            q = float(q_by_motor.get(motor_id, 0.0))
            if not math.isfinite(q):
                raise ValueError(f"{label}: motor {motor_id} target is not finite: {q}")
            if abs(q) > float(self.max_abs_joint_rad) + 1.0e-6:
                raise ValueError(
                    f"{label}: motor {motor_id} target {q:.6f} exceeds +/-{float(self.max_abs_joint_rad):.6f} rad"
                )
            lower = self.lower_limit_by_motor.get(motor_id, -float("inf"))
            upper = self.upper_limit_by_motor.get(motor_id, float("inf"))
            if q < lower - 1.0e-6 or q > upper + 1.0e-6:
                joint_name = self.motor_joint_map.get(motor_id, f"motor_{motor_id}")
                raise ValueError(
                    f"{label}: motor {motor_id} ({joint_name}) target {q:.6f} outside "
                    f"URDF limit [{lower:.6f}, {upper:.6f}]"
                )

        if self.collision_checker is not None and self.collision_checker.check(q_by_motor):
            pairs = self.collision_checker.colliding_pairs(q_by_motor)
            raise ValueError(f"{label}: self-collision detected pairs={pairs[:5]}")

        if self.check_gravity_load:
            tau_g = self.gravity.compute_gravity_by_motor(q_by_motor)
            for motor_id, limit in self.tau_limit_by_motor.items():
                if abs(float(tau_g[motor_id])) > float(limit):
                    raise ValueError(
                        f"{label}: gravity load too high motor {motor_id} "
                        f"|tau_g|={abs(float(tau_g[motor_id])):.3f}Nm > {float(limit):.3f}Nm"
                    )

    def _segment_collision_free(self, start: dict[int, float], end: dict[int, float]) -> bool:
        samples = self._sample_segment_qs(start, end, self.collision_samples_per_segment)
        if self.collision_checker is not None:
            any_collision, _idx = self.collision_checker.check_trajectory(samples)
            if any_collision:
                return False
        if self.check_gravity_load:
            for sample in samples:
                tau_g = self.gravity.compute_gravity_by_motor(sample)
                for motor_id, limit in self.tau_limit_by_motor.items():
                    if abs(float(tau_g[motor_id])) > float(limit):
                        return False
        return True

    def _sample_segment_qs(
        self,
        start: dict[int, float],
        end: dict[int, float],
        count: int,
    ) -> list[dict[int, float]]:
        out: list[dict[int, float]] = []
        n = max(2, int(count))
        for idx in range(n):
            u = idx / max(n - 1, 1)
            alpha = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            q = {}
            for motor_id in self.motor_ids:
                q0 = float(start.get(motor_id, 0.0))
                q1 = float(end.get(motor_id, q0))
                q[motor_id] = q0 + (q1 - q0) * alpha
            out.append(q)
        return out

    def _insert_zero_crossing_segments(self, segments: list[JointSegment]) -> list[JointSegment]:
        if not self.enforce_zero_crossing:
            return segments

        out: list[JointSegment] = []
        eps = float(self.zero_crossing_epsilon_rad)
        for segment in segments:
            crossing = [
                motor_id
                for motor_id in self.motor_ids
                if _sign_crosses_zero(
                    float(segment.start.get(motor_id, 0.0)),
                    float(segment.end.get(motor_id, 0.0)),
                    eps,
                )
            ]
            if not crossing:
                out.append(segment)
                continue

            mid = {}
            for motor_id in self.motor_ids:
                q0 = float(segment.start.get(motor_id, 0.0))
                q1 = float(segment.end.get(motor_id, q0))
                mid[motor_id] = 0.0 if motor_id in crossing else 0.5 * (q0 + q1)
            out.append(JointSegment(segment.start, mid, f"{segment.label}_to_zero"))
            out.append(JointSegment(mid, segment.end, f"{segment.label}_from_zero"))
            self.get_logger().warn(
                f"{segment.label}: sign crossing motors={crossing}; inserted explicit zero waypoint"
            )
        return out

    def _random_goal_xyz(self) -> np.ndarray:
        return self.workspace_center_xyz + self.rng.uniform(-self.workspace_delta_xyz, self.workspace_delta_xyz)

    def _sample_joint_fk_candidate(self, current: dict[int, float]) -> dict[int, float] | None:
        candidate = dict(current)
        for motor_id in self.controlled_motor_ids:
            max_delta = abs(float(self.joint_sample_delta_by_motor.get(motor_id, 0.16)))
            min_delta = min(abs(float(self.min_per_motion_joint_step_rad)), max_delta)
            sign = -1.0 if self.rng.random() < 0.5 else 1.0
            delta = sign * float(self.rng.uniform(min_delta, max_delta))
            lower = self.lower_limit_by_motor.get(motor_id, -float(self.max_abs_joint_rad))
            upper = self.upper_limit_by_motor.get(motor_id, float(self.max_abs_joint_rad))
            candidate[motor_id] = float(np.clip(candidate[motor_id] + delta, lower, upper))
        return candidate

    def _motion_requirement_satisfied(
        self,
        start: dict[int, float],
        end: dict[int, float],
    ) -> bool:
        min_step = abs(float(self.min_per_motion_joint_step_rad))
        for motor_id in self.require_motion_motor_ids:
            if abs(float(end[motor_id]) - float(start[motor_id])) < min_step:
                return False
        return True

    def _candidate_goal_xyz(self, current: dict[int, float]) -> tuple[np.ndarray, dict[int, float] | None]:
        mode = str(self.goal_sampling_mode).strip().lower()
        if mode in {"joint_fk", "joint-space", "joint_space"}:
            seed_candidate = self._sample_joint_fk_candidate(current)
            if seed_candidate is None:
                return self._random_goal_xyz(), None
            try:
                self._validate_q_or_raise(seed_candidate, "joint_fk_seed_candidate")
            except ValueError:
                return self._random_goal_xyz(), None
            goal_xyz = np.asarray(
                self.robot.forward_kinematics(seed_candidate, self.ik_solver.config.target_frame).translation,
                dtype=float,
            )
            return goal_xyz, seed_candidate
        if mode in {"workspace", "xyz", "cartesian"}:
            return self._random_goal_xyz(), None
        raise ValueError("goal_sampling_mode must be 'joint_fk' or 'workspace'")

    def _generate_ik_waypoints(self, start_q: dict[int, float]) -> list[JointSegment]:
        segments: list[JointSegment] = []
        current = dict(start_q)
        accepted = 0
        attempts = 0
        while accepted < int(self.random_waypoint_count) and attempts < int(self.max_generation_attempts):
            attempts += 1
            goal_xyz, _seed_candidate = self._candidate_goal_xyz(current)
            q_ref = self._ordered_controlled(current)
            result = self.ik_solver.solve_with_policy(
                goal_xyz,
                q_ref=q_ref,
                q_measured=q_ref,
                policy=self.ik_policy,
                rng=self.rng,
            )
            if not result.accepted or result.q_goal is None:
                continue

            candidate = self._full_with_controlled(current, result.q_goal)
            try:
                self._validate_q_or_raise(candidate, f"candidate_{accepted + 1:02d}")
            except ValueError:
                continue

            step_norm = float(
                np.linalg.norm(self._ordered_controlled(candidate) - self._ordered_controlled(current))
            )
            if step_norm < float(self.min_joint_step_norm_rad):
                continue
            if not self._motion_requirement_satisfied(current, candidate):
                continue
            if not self._segment_collision_free(current, candidate):
                continue

            accepted += 1
            segments.append(JointSegment(dict(current), candidate, f"ik_waypoint_{accepted:02d}"))
            current = candidate

        if accepted < int(self.random_waypoint_count):
            raise ValueError(
                f"failed to generate {self.random_waypoint_count} safe IK waypoints "
                f"after {attempts} attempts; try larger workspace_delta_xyz_json or looser IK limits"
            )
        self.get_logger().info(f"generated {accepted} safe IK waypoints after {attempts} attempts")
        return segments

    def _prepare_runtime_segments(self, raw_segments: list[JointSegment]) -> list[RuntimeSegment]:
        segments = self._insert_zero_crossing_segments(raw_segments)
        runtime: list[RuntimeSegment] = []
        v_max = np.full(len(self.motor_ids), float(self.v_max), dtype=float)
        a_max = np.full(len(self.motor_ids), float(self.a_max), dtype=float)
        zeros = np.zeros(len(self.motor_ids), dtype=float)

        for idx, segment in enumerate(segments):
            self._validate_q_or_raise(segment.end, f"{segment.label}.end")
            if not self._segment_collision_free(segment.start, segment.end):
                raise ValueError(f"{segment.label}: sampled segment is not collision/load safe")
            q0 = np.asarray([segment.start[motor_id] for motor_id in self.motor_ids], dtype=float)
            q1 = np.asarray([segment.end[motor_id] for motor_id in self.motor_ids], dtype=float)
            min_duration = self.initial_to_zero_duration_s if idx == 0 and "all_zero" in segment.label else self.segment_duration_s
            plan = plan_quintic(q0, q1, zeros, zeros, v_max, a_max, min_duration=min_duration)
            runtime.append(RuntimeSegment(segment=segment, plan=plan))
        return runtime

    def _initialize_sequence(self, now_s: float, connected_ids: list[int]) -> bool:
        if self.initialized:
            return True
        if not self.open_loop_preview:
            missing = [motor_id for motor_id in self.motor_ids if motor_id not in connected_ids]
            if missing:
                self.get_logger().warn(f"waiting for all motor states before dataset IK motion: missing={missing}")
                return False

        current_q = self._zero_q() if self.open_loop_preview else self._full_q_from_state()
        zero_q = self._zero_q()
        raw_segments: list[JointSegment] = []
        motion_start = current_q
        if self.start_all_motors_at_zero:
            raw_segments.append(JointSegment(dict(current_q), dict(zero_q), "current_to_all_zero"))
            motion_start = zero_q

        if self.move_to_workspace_center_first:
            self._validate_q_or_raise(self.workspace_center_q, "workspace_center_q")
            if not self._segment_collision_free(motion_start, self.workspace_center_q):
                raise ValueError("zero/current to workspace_center_q segment is not collision/load safe")
            raw_segments.append(
                JointSegment(dict(motion_start), dict(self.workspace_center_q), "zero_to_workspace_center")
            )
            motion_start = dict(self.workspace_center_q)

        raw_segments.extend(self._generate_ik_waypoints(motion_start))
        self.runtime_segments = self._prepare_runtime_segments(raw_segments)
        self.active_segment_idx = 0
        self.segment_started_s = now_s
        self.sequence_complete = False
        self.initialized = True
        self.last_segment_log_idx = -1
        self.get_logger().info(
            f"dataset IK motion sequence ready: segments={len(self.runtime_segments)} "
            f"first={self.runtime_segments[0].segment.label}"
        )
        return True

    def _sample_active(self, now_s: float) -> tuple[dict[int, float], dict[int, float]]:
        if not self.runtime_segments:
            return self.current_q_des_by_motor, self.current_qd_des_by_motor

        while self.active_segment_idx < len(self.runtime_segments):
            active = self.runtime_segments[self.active_segment_idx]
            elapsed_s = now_s - self.segment_started_s
            if elapsed_s <= active.plan.duration:
                break
            self.active_segment_idx += 1
            self.segment_started_s += active.plan.duration

        if self.active_segment_idx >= len(self.runtime_segments):
            if self.repeat:
                self.active_segment_idx = 0
                self.segment_started_s = now_s
                self.sequence_complete = False
                self.last_segment_log_idx = -1
            else:
                if not self.sequence_complete:
                    self.get_logger().info("dataset IK motion complete; holding final waypoint")
                    self.sequence_complete = True
                final = self.runtime_segments[-1].segment.end
                return dict(final), {motor_id: 0.0 for motor_id in self.motor_ids}

        active = self.runtime_segments[self.active_segment_idx]
        if self.active_segment_idx != self.last_segment_log_idx:
            self.last_segment_log_idx = self.active_segment_idx
            self.get_logger().info(
                f"segment {self.active_segment_idx + 1}/{len(self.runtime_segments)}: "
                f"{active.segment.label} duration={active.plan.duration:.2f}s"
            )

        q_vec, qd_vec, _done = sample_quintic(active.plan, now_s - self.segment_started_s)
        q_by_motor = {motor_id: float(q_vec[idx]) for idx, motor_id in enumerate(self.motor_ids)}
        qd_by_motor = {motor_id: float(qd_vec[idx]) for idx, motor_id in enumerate(self.motor_ids)}
        self.current_q_des_by_motor = q_by_motor
        self.current_qd_des_by_motor = qd_by_motor
        return q_by_motor, qd_by_motor

    def on_timer(self) -> None:
        now_s = time.monotonic()
        connected_ids = list(self.motor_ids) if self.open_loop_preview else self._connected_motor_ids(now_s)
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
        if not self._initialize_sequence(now_s, connected_ids):
            return
        q_des, qd_des = self._sample_active(now_s)
        self._publish_command(connected_ids, q_des, qd_des, now_s)

    def _publish_command(
        self,
        connected_ids: list[int],
        q_des_by_motor: dict[int, float],
        qd_des_by_motor: dict[int, float],
        now_s: float,
    ) -> None:
        q_measured = {}
        for motor_id in self.motor_ids:
            sample = self.state_by_motor[motor_id]
            if self.open_loop_preview:
                q_measured[motor_id] = float(q_des_by_motor.get(motor_id, 0.0))
            elif math.isfinite(sample.last_seen_s) and (now_s - sample.last_seen_s) <= self.state_timeout_s:
                q_measured[motor_id] = sample.q
            else:
                q_measured[motor_id] = q_des_by_motor.get(motor_id, 0.0)
        tau_g = self.gravity.compute_gravity_by_motor(q_measured)

        stamp = self.get_clock().now().to_msg()
        msg = MotorCMDArray()
        msg.stamp = stamp
        commands: list[MotorCMD] = []
        for motor_id in connected_ids:
            tuning = control_params_for_motor(motor_id)
            gravity_scale = _as_float_or_default(tuning.get("gravity_scale"), 1.0)
            gravity_bias = _as_float_or_default(tuning.get("gravity_bias"), 0.0)
            if self.use_tuning_gains:
                kp_value = _as_float_or_default(tuning.get("kp"), self.kp)
                kd_value = _as_float_or_default(tuning.get("kd"), self.kd)
            elif motor_id in self.controlled_motor_ids or self.start_all_motors_at_zero:
                kp_value = float(self.kp)
                kd_value = float(self.kd)
            else:
                kp_value = float(self.hold_kp)
                kd_value = float(self.hold_kd)

            tau_ff = gravity_scale * tau_g[motor_id] + gravity_bias
            tau_ff = _clip_symmetric(tau_ff, self.tau_limit_by_motor[motor_id])

            cmd = MotorCMD()
            cmd.stamp = stamp
            cmd.motor_id = int(motor_id)
            cmd.q_des = float(q_des_by_motor[motor_id])
            cmd.qd_des = float(qd_des_by_motor.get(motor_id, 0.0))
            cmd.kp = float(kp_value)
            cmd.kd = float(kd_value)
            cmd.tau_ff = float(tau_ff)
            commands.append(cmd)
        msg.commands = commands
        self.cmd_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DatasetIkMotionNode()
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
