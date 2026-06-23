"""Gravity-hold controller: /motor_state_array -> /motor_cmd_array."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Optional

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
from std_srvs.srv import Trigger

from phy.gravity import GravityCompensator


@dataclass
class MotorSample:
    q: float = 0.0
    qd: float = 0.0
    tau_measured: float = 0.0
    last_seen_s: float = float("-inf")


@dataclass
class CommandValues:
    q_des: float
    qd_des: float
    kp: float
    kd: float
    tau_ff: float


def _clip_symmetric(value: float, limit: float) -> float:
    if not math.isfinite(limit) or limit <= 0.0:
        return float(value)
    return float(max(-limit, min(limit, value)))


def _as_float_or_default(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


class HoldNode(Node):
    """Publish zero-gain commands with model-based gravity feedforward."""

    def __init__(self) -> None:
        super().__init__("hold_node")

        self.control_hz = declare_typed(self, "control_hz", 250.0)
        self.state_timeout_s = declare_typed(self, "state_timeout_s", 0.2)
        self.stale_warn_throttle_s = declare_typed(self, "stale_warn_throttle_s", 2.0)

        default_map_json = json.dumps(DEFAULT_MOTOR_JOINT_MAP)
        default_limit_json = json.dumps(DEFAULT_TAU_LIMIT_BY_MOTOR)
        map_json_text = declare_typed(self, "motor_joint_map_json", default_map_json)
        limit_json_text = declare_typed(self, "tau_limit_by_motor_json", default_limit_json)
        strip_str = lambda v: str(v).strip()
        urdf_path_text = declare_typed(self, "urdf_path", "", cast=strip_str)
        csv_log_path_text = declare_typed(self, "csv_log_path", "", cast=strip_str)

        motor_joint_map = parse_motor_joint_map_json(map_json_text)
        if not motor_joint_map:
            motor_joint_map = dict(DEFAULT_MOTOR_JOINT_MAP)
        tau_limit_map = parse_float_map_json(limit_json_text, "tau_limit_by_motor_json")
        if not tau_limit_map:
            tau_limit_map = dict(DEFAULT_TAU_LIMIT_BY_MOTOR)

        urdf_path = resolve_share_file("sim", "urdf/robot.urdf", urdf_path_text)
        self.gravity = GravityCompensator(urdf_path, motor_joint_map)

        self.motor_joint_map = motor_joint_map
        self.motor_ids = self.gravity.ordered_motor_ids
        self.tau_limit_by_motor = {
            motor_id: float(tau_limit_map.get(motor_id, float("inf"))) for motor_id in self.motor_ids
        }
        self.state_by_motor = {motor_id: MotorSample() for motor_id in self.motor_ids}
        self.last_cmd_by_motor: dict[int, CommandValues] = {}
        self.last_stale_warn_s = float("-inf")
        self.last_connected_ids: tuple[int, ...] = tuple()
        self._warn_times: dict[str, float] = {}  # throttled warn 타임스탬프

        self.csv_file: Optional[object] = None
        self.csv_writer: Optional[csv.writer] = None
        self.csv_path: Optional[Path] = None
        self._open_csv_logger(csv_log_path_text)

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

        self.halted = False
        self.halt_srv = self.create_service(Trigger, "/halt_robot", self._on_halt)
        self.resume_srv = self.create_service(Trigger, "/resume_robot", self._on_resume)

        self.get_logger().info(
            "hold_node initialized: "
            f"hz={self.control_hz:.1f} timeout={self.state_timeout_s:.3f}s "
            f"urdf={urdf_path} motors={list(self.motor_ids)} "
            f"csv={self.csv_path if self.csv_path is not None else 'disabled'}"
        )

    def _on_halt(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self.halted:
            self.halted = True
            self.get_logger().warn("/halt_robot activated — compliance-only (tuning bypassed)")
            response.message = "halted"
        else:
            response.message = "already halted"
        response.success = True
        return response

    def _on_resume(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.halted:
            self.halted = False
            self.get_logger().info("/resume_robot — gravity-comp + tuning restored")
            response.message = "resumed"
        else:
            response.message = "already running"
        response.success = True
        return response

    def _open_csv_logger(self, csv_log_path_text: str) -> None:
        if not csv_log_path_text:
            return
        csv_path = Path(csv_log_path_text).expanduser()
        if not csv_path.is_absolute():
            csv_path = (Path.cwd() / csv_path).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        existed_and_nonempty = csv_path.exists() and csv_path.stat().st_size > 0
        csv_file = csv_path.open("a", encoding="utf-8", newline="")
        writer = csv.writer(csv_file)
        if not existed_and_nonempty:
            writer.writerow(
                [
                    "timestamp",
                    "motor_id",
                    "q",
                    "qd",
                    "tau_meas",
                    "tau_cmd",
                    "tau_g_model",
                    "gravity_scale",
                    "gravity_bias",
                ]
            )
            csv_file.flush()
        self.csv_file = csv_file
        self.csv_writer = writer
        self.csv_path = csv_path

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

    def on_timer(self) -> None:
        now_s = time.monotonic()
        connected_ids = self._connected_motor_ids(now_s)
        connected_tuple = tuple(connected_ids)
        if connected_tuple != self.last_connected_ids:
            self.last_connected_ids = connected_tuple
            self.get_logger().info(f"connected motors updated: {list(connected_tuple)}")

        if connected_ids:
            self._publish_gravity_hold(connected_ids, now_s)
            return

        if (now_s - self.last_stale_warn_s) >= self.stale_warn_throttle_s:
            self.get_logger().warn(
                f"no connected motor state within timeout ({self.state_timeout_s:.3f}s): skip publish"
            )
            self.last_stale_warn_s = now_s
        self.last_cmd_by_motor = {}

    def _warn_throttle(self, key: str, msg: str, now_s: float, interval_s: float = 2.0) -> None:
        if now_s - self._warn_times.get(key, float("-inf")) >= interval_s:
            self._warn_times[key] = now_s
            self.get_logger().warn(msg)

    def _publish_gravity_hold(self, connected_ids: list[int], now_s: float) -> None:
        q_by_motor = {}
        for motor_id in self.motor_ids:
            sample = self.state_by_motor[motor_id]
            if math.isfinite(sample.last_seen_s) and (now_s - sample.last_seen_s) <= self.state_timeout_s:
                q_by_motor[motor_id] = sample.q
            else:
                q_by_motor[motor_id] = 0.0
        tau_g_by_motor = self.gravity.compute_gravity_by_motor(q_by_motor)

        cmd_values: dict[int, CommandValues] = {}
        scale_by_motor: dict[int, float] = {}
        bias_by_motor: dict[int, float] = {}
        for motor_id in connected_ids:
            sample = self.state_by_motor[motor_id]
            if self.halted:
                gravity_scale = 1.0
                gravity_bias = 0.0
            else:
                tuning = control_params_for_motor(motor_id)
                gravity_scale = _as_float_or_default(tuning.get("gravity_scale"), 1.0)
                gravity_bias = _as_float_or_default(tuning.get("gravity_bias"), 0.0)
            tau_raw = gravity_scale * tau_g_by_motor[motor_id] + gravity_bias
            tau_cmd = _clip_symmetric(tau_raw, self.tau_limit_by_motor[motor_id])
            if abs(tau_cmd - tau_raw) > 1e-4:
                self._warn_throttle(
                    f"tau_clamp_{motor_id}",
                    f"[SAFETY] motor {motor_id} tau_ff clamped: "
                    f"{tau_raw:.3f} → {tau_cmd:.3f} Nm "
                    f"(limit=±{self.tau_limit_by_motor[motor_id]:.1f})",
                    now_s,
                )

            cmd_values[motor_id] = CommandValues(
                q_des=sample.q,
                qd_des=0.0,
                kp=0.0,
                kd=0.0,
                tau_ff=tau_cmd,
            )
            scale_by_motor[motor_id] = gravity_scale
            bias_by_motor[motor_id] = gravity_bias

        stamp_s = self._publish_command_values(cmd_values)
        self.last_cmd_by_motor = cmd_values
        self._append_csv_rows(stamp_s, cmd_values, tau_g_by_motor, scale_by_motor, bias_by_motor)

    def _publish_command_values(self, cmd_values: dict[int, CommandValues]) -> float:
        stamp = self.get_clock().now().to_msg()
        msg = MotorCMDArray()
        msg.stamp = stamp
        commands = []
        for motor_id in sorted(cmd_values.keys()):
            values = cmd_values[motor_id]
            cmd = MotorCMD()
            cmd.stamp = stamp
            cmd.motor_id = int(motor_id)
            cmd.q_des = float(values.q_des)
            cmd.qd_des = float(values.qd_des)
            cmd.kp = float(values.kp)
            cmd.kd = float(values.kd)
            cmd.tau_ff = float(values.tau_ff)
            commands.append(cmd)
        msg.commands = commands
        self.cmd_pub.publish(msg)
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

    def _append_csv_rows(
        self,
        stamp_s: float,
        cmd_values: dict[int, CommandValues],
        tau_g_by_motor: dict[int, float],
        scale_by_motor: dict[int, float],
        bias_by_motor: dict[int, float],
    ) -> None:
        if self.csv_writer is None or self.csv_file is None:
            return
        for motor_id in sorted(cmd_values.keys()):
            sample = self.state_by_motor[motor_id]
            values = cmd_values[motor_id]
            self.csv_writer.writerow(
                [
                    f"{stamp_s:.9f}",
                    int(motor_id),
                    f"{sample.q:.9f}",
                    f"{sample.qd:.9f}",
                    f"{sample.tau_measured:.9f}",
                    f"{values.tau_ff:.9f}",
                    f"{tau_g_by_motor[motor_id]:.9f}",
                    f"{scale_by_motor[motor_id]:.9f}",
                    f"{bias_by_motor[motor_id]:.9f}",
                ]
            )

    def destroy_node(self) -> bool:
        if self.csv_file is not None:
            try:
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = None
            self.csv_writer = None
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HoldNode()
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
