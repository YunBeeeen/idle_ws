"""Terminal-driven Cartesian target tracking with gravity-compensated joint commands.

Usage example:
  ros2 run phy ee_xyz_trajectory_node
  target xyz> 0.20 0.00 0.35
  target xyz> (0.18, -0.05, 0.30)
  ros2 topic pub /ee_target_xyz std_msgs/msg/Float64MultiArray "{data: [0.20, 0.00, 0.35]}" -r 10
  ros2 topic pub /ee_target_point geometry_msgs/msg/Point "{x: 0.20, y: 0.00, z: 0.35}" -r 10
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import queue
import re
import sys
import threading
import time
from typing import Optional, TextIO

from geometry_msgs.msg import Point
import numpy as np
import rclpy
from idle_common.control_tuning import control_params_for_motor
from idle_common.paths import resolve_share_file
from idle_common.ros_params import declare_typed
from msgs.msg import MotorCMD, MotorCMDArray, MotorStateArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray

from idle_common.motor_map import (
    DEFAULT_MOTOR_JOINT_MAP,
    DEFAULT_TAU_LIMIT_BY_MOTOR,
    parse_float_map_json,
    parse_motor_joint_map_json,
)
from phy.gravity import GravityCompensator
from phy.ik import IKConfig, IKPolicyConfig, IKSolver
from phy.traj import QuinticPlan, plan_quintic, sample_quintic


@dataclass
class MotorSample:
    q: float = 0.0
    qd: float = 0.0
    tau_measured: float = 0.0
    last_seen_s: float = float("-inf")


@dataclass
class TrajectoryRuntime:
    plan: QuinticPlan
    start_s: float
    goal_xyz: np.ndarray


def _clip_symmetric(value: float, limit: float) -> float:
    if not math.isfinite(limit) or limit <= 0.0:
        return float(value)
    return float(max(-limit, min(limit, value)))


def _as_float_or_default(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _parse_xyz_line(text: str) -> np.ndarray:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty input")
    if stripped.startswith("(") and stripped.endswith(")"):
        stripped = stripped[1:-1].strip()
    number_tokens = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", stripped)
    if len(number_tokens) < 3:
        raise ValueError("expected 3 numeric values: x y z")
    if len(number_tokens) > 3:
        number_tokens = number_tokens[-3:]
    xyz = [float(tok) for tok in number_tokens]
    return np.asarray(xyz, dtype=float)


def _parse_int_list_json(text: str, field_name: str) -> list[int]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON list: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a JSON list")
    out: list[int] = []
    for idx, value in enumerate(raw):
        try:
            out.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{idx}] must be an integer") from exc
    return out


def _parse_float_tuple3_json(text: str, field_name: str) -> tuple[float, float, float]:
    stripped = text.strip()
    if not stripped:
        return (0.0, 0.0, 0.0)
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON list: {exc}") from exc
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{field_name} must be a JSON list of length 3")
    if not all(isinstance(v, (int, float)) for v in raw):
        raise ValueError(f"{field_name} values must be numeric")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


class EEXyzTrajectoryNode(Node):
    """Track terminal Cartesian targets with IK + quintic trajectory + gravity feedforward."""

    def __init__(self) -> None:
        super().__init__("ee_xyz_trajectory_node")

        strip_str = lambda v: str(v).strip()

        self.control_hz = declare_typed(self, "control_hz", 250.0)
        self.state_timeout_s = declare_typed(self, "state_timeout_s", 0.2)
        self.stale_warn_throttle_s = declare_typed(self, "stale_warn_throttle_s", 2.0)

        self.kp = declare_typed(self, "kp", 1.0)
        self.kd = declare_typed(self, "kd", 0.05)
        self.min_traj_duration = declare_typed(self, "min_traj_duration", 0.2)
        self.v_max = declare_typed(self, "v_max", 0.8)
        self.a_max = declare_typed(self, "a_max", 1.5)
        self.ee_log_hz = declare_typed(self, "ee_log_hz", 2.0)
        self.target_dedup_epsilon_m = declare_typed(self, "target_dedup_epsilon_m", 1.0e-4)
        self.max_ik_residual_accept_m = declare_typed(self, "max_ik_residual_accept_m", 0.005)
        self.ik_random_restarts = declare_typed(self, "ik_random_restarts", 24)
        self.ik_seed_default_span = declare_typed(self, "ik_seed_default_span", math.pi)
        # <=0 disables jump-based rejection to allow long detours when cable routing requires it.
        self.max_joint_jump_rad = declare_typed(self, "max_joint_jump_rad", 0.0)
        self.reset_on_controlled_disconnect = declare_typed(self, "reset_on_controlled_disconnect", True)

        self.enable_terminal_input = declare_typed(self, "enable_terminal_input", True)
        self.input_prompt = declare_typed(self, "input_prompt", "target xyz> ")
        self.target_topic = declare_typed(self, "target_topic", "/ee_target_xyz", cast=strip_str)
        if not self.target_topic:
            self.target_topic = "/ee_target_xyz"
        self.target_point_topic = declare_typed(self, "target_point_topic", "/ee_target_point", cast=strip_str)
        if not self.target_point_topic:
            self.target_point_topic = "/ee_target_point"

        default_map_json = json.dumps(DEFAULT_MOTOR_JOINT_MAP)
        default_limit_json = json.dumps(DEFAULT_TAU_LIMIT_BY_MOTOR)
        map_json_text = declare_typed(self, "motor_joint_map_json", default_map_json)
        limit_json_text = declare_typed(self, "tau_limit_by_motor_json", default_limit_json)
        controlled_ids_json = declare_typed(self, "controlled_motor_ids_json", "[1,2,3]")
        urdf_path_text = declare_typed(self, "urdf_path", "", cast=strip_str)

        target_frame = declare_typed(self, "target_frame", "ee_link", cast=strip_str)
        target_offset_json = declare_typed(self, "target_offset_xyz_json", "[0.0,0.0,0.0]")
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

        controlled_ids = _parse_int_list_json(controlled_ids_json, "controlled_motor_ids_json")
        if not controlled_ids:
            controlled_ids = [1, 2, 3]
        controlled_ids = sorted(set(controlled_ids))
        missing_ids = [motor_id for motor_id in controlled_ids if motor_id not in motor_joint_map]
        if missing_ids:
            raise ValueError(
                f"controlled motors missing in motor_joint_map_json: {missing_ids}; "
                f"map keys={sorted(motor_joint_map.keys())}"
            )

        urdf_path = resolve_share_file("sim", "urdf/robot.urdf", urdf_path_text)
        self.gravity = GravityCompensator(urdf_path, motor_joint_map)
        self.motor_joint_map = motor_joint_map
        self.motor_ids = self.gravity.ordered_motor_ids
        self.controlled_motor_ids = tuple(controlled_ids)
        self.controlled_joint_names = tuple(self.motor_joint_map[motor_id] for motor_id in self.controlled_motor_ids)

        target_offset = _parse_float_tuple3_json(target_offset_json, "target_offset_xyz_json")
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
        self.ik_policy_config = IKPolicyConfig(
            max_ik_residual_accept_m=self.max_ik_residual_accept_m,
            ik_random_restarts=self.ik_random_restarts,
            ik_seed_default_span=self.ik_seed_default_span,
            max_joint_jump_rad=self.max_joint_jump_rad,
            use_heuristic_seed=True,
        )

        dof = len(self.controlled_motor_ids)
        self.v_max_vec = np.full(dof, max(self.v_max, 1.0e-3), dtype=float)
        self.a_max_vec = np.full(dof, max(self.a_max, 1.0e-3), dtype=float)

        self.tau_limit_by_motor = {
            motor_id: float(tau_limit_map.get(motor_id, float("inf"))) for motor_id in self.motor_ids
        }
        self.state_by_motor = {motor_id: MotorSample() for motor_id in self.motor_ids}
        self.last_stale_warn_s = float("-inf")
        self.last_connected_ids: tuple[int, ...] = tuple()

        self.ctrl_q_des = np.zeros(dof, dtype=float)
        self.ctrl_qd_des = np.zeros(dof, dtype=float)
        self.ctrl_initialized = False
        self.active_traj: Optional[TrajectoryRuntime] = None
        self.last_goal_xyz: Optional[np.ndarray] = None
        self.last_requested_goal_xyz: Optional[np.ndarray] = None
        self.last_ee_log_s = float("-inf")
        self.last_ignored_target_warn_s = float("-inf")
        self.rng = np.random.default_rng()

        self.goal_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.stdin_stop_event = threading.Event()
        self.stdin_thread: Optional[threading.Thread] = None
        self.input_stream: Optional[TextIO] = None
        self.input_stream_is_tty = False
        self.input_stream_source = "disabled"

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
        qos_target = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.target_sub = self.create_subscription(
            Float64MultiArray, self.target_topic, self.on_target_xyz, qos_target
        )
        self.target_point_sub = self.create_subscription(
            Point, self.target_point_topic, self.on_target_point, qos_target
        )
        self.cmd_pub = self.create_publisher(MotorCMDArray, "/motor_cmd_array", qos_cmd)

        period_s = max(1.0 / self.control_hz, 1.0e-4)
        self.control_timer = self.create_timer(period_s, self.on_timer)
        if self.enable_terminal_input:
            self._start_stdin_thread()
        else:
            self.get_logger().info("terminal input disabled by parameter")

        self.get_logger().info(
            "ee_xyz_trajectory_node initialized: "
            f"hz={self.control_hz:.1f} kp={self.kp:.3f} kd={self.kd:.3f} "
            f"ee_log_hz={self.ee_log_hz:.2f} "
            f"target_dedup_eps={self.target_dedup_epsilon_m:.6f}m "
            f"ik_accept_residual<={self.max_ik_residual_accept_m:.3f}m "
            f"ik_random_restarts={self.ik_random_restarts} "
            f"max_joint_jump_rad="
            f"{'disabled' if self.max_joint_jump_rad <= 0.0 else f'{self.max_joint_jump_rad:.3f}'} "
            f"target_topic={self.target_topic} point_topic={self.target_point_topic} "
            f"target_frame={target_frame} joints={list(self.controlled_joint_names)} "
            f"motors={list(self.motor_ids)} urdf={urdf_path}"
        )

    def _start_stdin_thread(self) -> None:
        stream = self._resolve_input_stream()
        if stream is None:
            self.get_logger().warn("terminal input unavailable: neither stdin nor /dev/tty is usable")
            return
        self.input_stream = stream
        self.input_stream_is_tty = bool(getattr(stream, "isatty", lambda: False)())
        self.stdin_thread = threading.Thread(target=self._stdin_loop, name="ee_xyz_stdin", daemon=True)
        self.stdin_thread.start()
        self.get_logger().info(
            "terminal input enabled (x y z | x,y,z | (x,y,z)); "
            f"source={self.input_stream_source}"
        )

    def _resolve_input_stream(self) -> Optional[TextIO]:
        if sys.stdin is not None and bool(getattr(sys.stdin, "isatty", lambda: False)()):
            self.input_stream_source = "stdin"
            return sys.stdin
        try:
            tty_stream = open("/dev/tty", "r", encoding="utf-8", errors="replace")
        except Exception:
            return None
        self.input_stream_source = "/dev/tty"
        return tty_stream

    def _stdin_loop(self) -> None:
        stream = self.input_stream
        if stream is None:
            return
        while not self.stdin_stop_event.is_set():
            try:
                if self.input_stream_is_tty and self.input_prompt:
                    print(self.input_prompt, end="", flush=True)
                line = stream.readline()
            except Exception as exc:
                self.get_logger().warn(f"stdin read failed: {exc}")
                return

            if line == "":
                if self.stdin_stop_event.is_set():
                    return
                if not self.input_stream_is_tty:
                    self.get_logger().warn("terminal input EOF reached: input thread stopped")
                    return
                if self.input_stream_source == "/dev/tty":
                    self.get_logger().warn("/dev/tty EOF reached: input thread stopped")
                    return
                time.sleep(0.05)
                continue

            text = line.strip()
            if not text:
                continue
            if text.lower() in {"q", "quit", "exit"}:
                self.get_logger().info("ignore quit token; controller keeps running")
                continue

            try:
                goal_xyz = _parse_xyz_line(text)
            except ValueError as exc:
                self.get_logger().warn(f"invalid target '{text}': {exc}")
                continue
            self._queue_goal(goal_xyz, source=self.input_stream_source)

    def _is_duplicate_goal(self, goal_xyz: np.ndarray) -> bool:
        ref_goal: Optional[np.ndarray] = None
        if self.active_traj is not None:
            ref_goal = self.active_traj.goal_xyz
        elif self.last_goal_xyz is not None:
            ref_goal = self.last_goal_xyz
        elif self.last_requested_goal_xyz is not None:
            ref_goal = self.last_requested_goal_xyz
        if ref_goal is None:
            return False
        distance = float(np.linalg.norm(goal_xyz - ref_goal))
        return distance <= max(self.target_dedup_epsilon_m, 0.0)

    def _queue_goal(self, goal_xyz: np.ndarray, source: str) -> None:
        if self._is_duplicate_goal(goal_xyz):
            now_s = time.monotonic()
            if (now_s - self.last_ignored_target_warn_s) >= 1.0:
                self.last_ignored_target_warn_s = now_s
                self.get_logger().info(
                    f"ignored duplicate target ({source}) xyz=({goal_xyz[0]:.4f}, {goal_xyz[1]:.4f}, {goal_xyz[2]:.4f})"
                )
            return
        self.last_requested_goal_xyz = goal_xyz.copy()
        try:
            while True:
                self.goal_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.goal_queue.put_nowait(goal_xyz)
        except queue.Full:
            pass
        self.get_logger().info(
            f"queued target ({source}) xyz=({goal_xyz[0]:.4f}, {goal_xyz[1]:.4f}, {goal_xyz[2]:.4f})"
        )

    def on_target_xyz(self, msg: Float64MultiArray) -> None:
        raw = list(msg.data)
        if len(raw) < 3:
            self.get_logger().warn(
                f"ignored target topic message on {self.target_topic}: need 3 values, got {len(raw)}"
            )
            return
        goal_xyz = np.asarray([raw[0], raw[1], raw[2]], dtype=float)
        if not np.all(np.isfinite(goal_xyz)):
            self.get_logger().warn(f"ignored target topic message on {self.target_topic}: non-finite values")
            return
        self._queue_goal(goal_xyz, source=f"topic:{self.target_topic}")

    def on_target_point(self, msg: Point) -> None:
        goal_xyz = np.asarray([msg.x, msg.y, msg.z], dtype=float)
        if not np.all(np.isfinite(goal_xyz)):
            self.get_logger().warn(
                f"ignored target topic message on {self.target_point_topic}: non-finite values"
            )
            return
        self._queue_goal(goal_xyz, source=f"topic:{self.target_point_topic}")

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
            if not math.isfinite(sample.last_seen_s):
                continue
            if (now_s - sample.last_seen_s) <= self.state_timeout_s:
                connected.append(motor_id)
        return connected

    def _consume_latest_goal(self) -> Optional[np.ndarray]:
        latest: Optional[np.ndarray] = None
        while True:
            try:
                latest = self.goal_queue.get_nowait()
            except queue.Empty:
                return latest

    def _measured_ctrl_q(self, now_s: float) -> Optional[np.ndarray]:
        q: list[float] = []
        for motor_id in self.controlled_motor_ids:
            sample = self.state_by_motor[motor_id]
            if not math.isfinite(sample.last_seen_s):
                return None
            if (now_s - sample.last_seen_s) > self.state_timeout_s:
                return None
            q.append(sample.q)
        return np.asarray(q, dtype=float)

    def _ensure_ctrl_initialized(self, now_s: float) -> bool:
        if self.ctrl_initialized:
            return True
        measured = self._measured_ctrl_q(now_s)
        if measured is None:
            return False
        self.ctrl_q_des = measured.copy()
        self.ctrl_qd_des = np.zeros_like(measured)
        self.ctrl_initialized = True
        return True

    def _update_desired_from_active_traj(self, now_s: float) -> None:
        if self.active_traj is None:
            return
        elapsed_s = now_s - self.active_traj.start_s
        q_des, qd_des, done = sample_quintic(self.active_traj.plan, elapsed_s)
        self.ctrl_q_des = q_des
        self.ctrl_qd_des = qd_des
        if done:
            self.ctrl_qd_des = np.zeros_like(self.ctrl_q_des)
            self.active_traj = None

    def _start_trajectory_for_goal(self, goal_xyz: np.ndarray, now_s: float) -> None:
        if not self._ensure_ctrl_initialized(now_s):
            self.get_logger().warn(
                "cannot solve IK yet: waiting for fresh states of controlled motors "
                f"{list(self.controlled_motor_ids)}"
            )
            return

        self._update_desired_from_active_traj(now_s)
        q_start = self.ctrl_q_des.copy()
        qd_start = self.ctrl_qd_des.copy()
        measured_q = self._measured_ctrl_q(now_s)
        policy_result = self.ik_solver.solve_with_policy(
            goal_xyz=goal_xyz,
            q_ref=q_start,
            q_measured=measured_q,
            policy=self.ik_policy_config,
            rng=self.rng,
        )
        if not policy_result.accepted:
            if policy_result.reason == "rejected_residual":
                self.get_logger().warn(
                    f"target rejected: IK residual too large ({policy_result.best_residual:.6f}m > "
                    f"{self.max_ik_residual_accept_m:.6f}m) xyz=({goal_xyz[0]:.4f}, {goal_xyz[1]:.4f}, {goal_xyz[2]:.4f})"
                )
                return
            if policy_result.reason == "rejected_joint_jump":
                self.get_logger().warn(
                    f"target rejected: joint jump too large ({policy_result.best_jump_rad:.3f}rad > "
                    f"{self.max_joint_jump_rad:.3f}rad) "
                    f"xyz=({goal_xyz[0]:.4f}, {goal_xyz[1]:.4f}, {goal_xyz[2]:.4f})"
                )
                return
            self.get_logger().warn(
                "target rejected: unknown IK policy reason "
                f"'{policy_result.reason}' xyz=({goal_xyz[0]:.4f}, {goal_xyz[1]:.4f}, {goal_xyz[2]:.4f})"
            )
            return
        if policy_result.q_goal is None:
            self.get_logger().warn(
                f"target rejected: IK policy accepted without q_goal xyz=({goal_xyz[0]:.4f}, "
                f"{goal_xyz[1]:.4f}, {goal_xyz[2]:.4f})"
            )
            return

        ik = policy_result.best_result
        q_goal = policy_result.q_goal
        plan = plan_quintic(
            q_start=q_start,
            q_goal=q_goal,
            v_start=qd_start,
            v_goal=np.zeros_like(q_goal),
            v_max=self.v_max_vec,
            a_max=self.a_max_vec,
            min_duration=max(self.min_traj_duration, 1.0e-3),
        )
        self.active_traj = TrajectoryRuntime(plan=plan, start_s=now_s, goal_xyz=goal_xyz.copy())
        self.last_goal_xyz = goal_xyz.copy()
        predicted_xyz = self.ik_solver.forward_position(q_goal)
        msg = (
            f"new target accepted xyz=({goal_xyz[0]:.4f}, {goal_xyz[1]:.4f}, {goal_xyz[2]:.4f}) "
            f"traj={plan.duration:.3f}s residual={policy_result.best_residual:.6f} "
            f"predicted=({predicted_xyz[0]:.4f}, {predicted_xyz[1]:.4f}, {predicted_xyz[2]:.4f})"
        )
        if ik.success:
            self.get_logger().info(msg)
        else:
            self.get_logger().warn(f"{msg} (IK did not meet tolerance in {ik.iterations} iters)")

    def _log_ee_pose(self, now_s: float) -> None:
        if self.ee_log_hz <= 0.0:
            return
        period_s = 1.0 / max(self.ee_log_hz, 1.0e-6)
        if (now_s - self.last_ee_log_s) < period_s:
            return
        self.last_ee_log_s = now_s

        measured_q = self._measured_ctrl_q(now_s)
        if measured_q is None:
            self.get_logger().info("ee_link pose unavailable: waiting for fresh /motor_state_array")
            return
        ee_xyz = self.ik_solver.forward_position(measured_q)
        log = f"ee_link now=({ee_xyz[0]:.4f}, {ee_xyz[1]:.4f}, {ee_xyz[2]:.4f})"
        if self.ctrl_initialized:
            des_xyz = self.ik_solver.forward_position(self.ctrl_q_des)
            log += f" des=({des_xyz[0]:.4f}, {des_xyz[1]:.4f}, {des_xyz[2]:.4f})"
        goal_xyz = self.active_traj.goal_xyz if self.active_traj is not None else self.last_goal_xyz
        if goal_xyz is not None:
            log += f" goal=({goal_xyz[0]:.4f}, {goal_xyz[1]:.4f}, {goal_xyz[2]:.4f})"
        self.get_logger().info(log)

    def on_timer(self) -> None:
        now_s = time.monotonic()
        prev_connected_set = set(self.last_connected_ids)
        connected_ids = self._connected_motor_ids(now_s)
        connected_set = set(connected_ids)
        connected_tuple = tuple(connected_ids)
        if connected_tuple != self.last_connected_ids:
            self.last_connected_ids = connected_tuple
            self.get_logger().info(f"connected motors updated: {list(connected_tuple)}")
            if self.reset_on_controlled_disconnect:
                lost_controlled = [
                    motor_id
                    for motor_id in self.controlled_motor_ids
                    if motor_id in prev_connected_set and motor_id not in connected_set
                ]
                if lost_controlled:
                    self.active_traj = None
                    self.ctrl_initialized = False
                    self.ctrl_qd_des = np.zeros_like(self.ctrl_qd_des)
                    self.get_logger().warn(
                        f"controlled motors disconnected {lost_controlled}: reset trajectory/desired state"
                    )

        latest_goal = self._consume_latest_goal()
        if latest_goal is not None:
            self._start_trajectory_for_goal(latest_goal, now_s)
        self._log_ee_pose(now_s)

        if not connected_ids:
            if (now_s - self.last_stale_warn_s) >= self.stale_warn_throttle_s:
                self.get_logger().warn(
                    f"no connected motor state within timeout ({self.state_timeout_s:.3f}s): skip publish"
                )
                self.last_stale_warn_s = now_s
            return

        if not self._ensure_ctrl_initialized(now_s):
            return
        self._update_desired_from_active_traj(now_s)

        q_des_by_motor: dict[int, float] = {}
        qd_des_by_motor: dict[int, float] = {}
        for idx, motor_id in enumerate(self.controlled_motor_ids):
            q_des_by_motor[motor_id] = float(self.ctrl_q_des[idx])
            qd_des_by_motor[motor_id] = float(self.ctrl_qd_des[idx])

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
            sample = self.state_by_motor[motor_id]
            is_ctrl = motor_id in q_des_by_motor
            q_des = q_des_by_motor[motor_id] if is_ctrl else sample.q
            qd_des = qd_des_by_motor[motor_id] if is_ctrl else 0.0

            tuning = control_params_for_motor(motor_id)
            gravity_scale = _as_float_or_default(tuning.get("gravity_scale"), 1.0)
            gravity_bias = _as_float_or_default(tuning.get("gravity_bias"), 0.0)
            kp_value = _as_float_or_default(tuning.get("kp"), self.kp) if is_ctrl else 0.0
            kd_value = _as_float_or_default(tuning.get("kd"), self.kd) if is_ctrl else 0.0
            tau_ff = gravity_scale * tau_g_by_motor[motor_id] + gravity_bias
            tau_ff = _clip_symmetric(tau_ff, self.tau_limit_by_motor[motor_id])

            cmd = MotorCMD()
            cmd.stamp = stamp
            cmd.motor_id = int(motor_id)
            cmd.q_des = float(q_des)
            cmd.qd_des = float(qd_des)
            cmd.kp = float(kp_value)
            cmd.kd = float(kd_value)
            cmd.tau_ff = float(tau_ff)
            commands.append(cmd)
        msg.commands = commands
        self.cmd_pub.publish(msg)

    def destroy_node(self) -> bool:
        self.stdin_stop_event.set()
        if self.input_stream is not None and self.input_stream is not sys.stdin:
            try:
                self.input_stream.close()
            except Exception:
                pass
            self.input_stream = None
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = EEXyzTrajectoryNode()
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
